from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import (
    Dict,
    Type,
    Generic,
    TypeVar,
    Any,
    Sequence,
    Iterator,
    List,
    Tuple,
    Optional,
    Callable,
    Awaitable,
)

from wrangle_pypes.pipeline import TransformationException, Transformation, Chain

__all__ = ("AsyncPipeline",)


M = TypeVar("M", covariant=True)


@dataclass
class AsyncPipeline(Generic[M]):
    transformations: Dict[Type[M], Dict[str, AsyncTransformation]] = field(
        default_factory=dict
    )
    lookup: Optional[Callable[[Type[M], Dict[str, Any]], Awaitable[M]]] = None

    async def create(self, model: Type[M], data: Any, *args, **kwargs) -> Awaitable[M]:
        """Build a single instance of given model.
        
        :param model: model to build
        :param data: data to build instance from
        """
        return await model(  # type: ignore
            **self.build_kwargs(model, data, *args, **kwargs)
        )

    async def create_multiple(
        self, model: Type[M], data: Sequence[Any], *args, **kwargs
    ) -> Iterator[Awaitable[M]]:
        """Build multiple instance of given model.
        
        :param model: model to build
        :param data: data to build instances from
        """
        return (
            await self.create(model, datapoint, *args, **kwargs) for datapoint in data
        )

    async def get_or_create(
        self,
        model: Type[M],
        data: Any,
        match_targets: List[str],
        *args,
        lookup: Optional[Callable[[Type[M], Dict[str, Any]], Awaitable[M]]] = None,
        **kwargs,
    ) -> Tuple[M, bool]:
        """Get a single instance matching given match targets or create it.
        
        :param model: model to get or create the instance for
        :param data: data to get or create the instance from
        :param match_targets: fields to use for matching the data to the DB, default to None
        """
        lookup = lookup or getattr(self, "lookup")
        if lookup is None:
            raise NameError("Need to supply lookup to use `get or create` features.")

        if not match_targets:
            lookup_kwargs = await self.build_kwargs(model, data, *args, **kwargs)
        else:
            lookup_kwargs = {
                key: await self.build_kwarg(model, key, data, *args, **kwargs)
                for key in match_targets
            }

        instance = await lookup(model, lookup_kwargs)
        if not instance:
            if not match_targets:
                build_kwargs = await self.build_kwargs(model, data, *args, **kwargs)
            return model(**build_kwargs), True  # type: ignore
        return instance, False

    async def get_or_create_multiple(
        self,
        model: Type[M],
        data: Sequence[Any],
        match_targets: List[str],
        *args,
        lookup: Optional[Callable[[Type[M], Dict[str, Any]], Awaitable[M]]] = None,
        **kwargs,
    ) -> Iterator[Tuple[M, bool]]:
        """Get a instances matching given match targets or create them.
        
        :param model: model to get or create the instances for
        :param data: data to get or create the instances from
        :param match_targets: fields to use for matching the data to the DB, default to None
        """
        return (
            await self.get_or_create(
                model, datapoint, match_targets, *args, lookup=lookup, **kwargs
            )
            for datapoint in data
        )

    async def build_kwargs(
        self, model: Type[M], data: Any, *args, **kwargs
    ) -> Dict[str, Any]:
        """Build kwargs for an instance of given model.
        
        :param model: model to build kwargs for
        :param data: data to build kwargs from
        """
        return {
            name: await self.build_kwarg(model, name, data, *args, **kwargs)
            for name, transformation in self.transformations[model].items()
        }

    async def build_kwarg(
        self, model: Type[M], kwarg: str, data: Any, *args, **kwargs
    ) -> Dict[str, Any]:
        """Build a single kwarg for an instance of given model.
        
        :param model: model to build kwarg for
        :param kwarg: kwargs to build
        :param data: data to build kwarg from
        """
        try:
            return await self.transformations[model][kwarg](self, data, *args, **kwargs)
        except TransformationException as e:
            e_type, transformation, *e_args = e.args
            raise e_type(
                f"failed @ {model.__name__}.{kwarg}: {transformation.__name__}: {e_args[0]}",
                *e_args[1:],
            )


@dataclass
class AsyncTransformation:
    async def apply(self, pipeline: AsyncPipeline, data: Any, *args, **kwargs) -> Any:
        raise NotImplementedError

    async def __call__(
        self, pipeline: AsyncPipeline, data: Any, *args, **kwargs
    ) -> Any:
        try:
            return await self.apply(pipeline, data, *args, **kwargs)
        except Exception as e:
            if isinstance(e, TransformationException):
                raise e
            raise TransformationException(type(e), type(self), *e.args)

    def __or__(self, other: AsyncTransformation) -> AsyncChain:
        return AsyncChain() | self | other


@dataclass
class AsyncChain(AsyncTransformation):
    transformations: List[AsyncTransformation] = field(init=False, default_factory=list)

    async def apply(self, pipeline: AsyncPipeline, data: Any, *args, **kwargs) -> Any:
        val = data
        for transformation in self.transformations:
            val = await transformation(pipeline, val, *args, **kwargs)
        return val

    def __or__(self, other: AsyncTransformation) -> AsyncChain:
        self.transformations.append(other)
        return self

