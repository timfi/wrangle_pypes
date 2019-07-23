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
)


__all__ = ("Pipeline",)


M = TypeVar("M", covariant=True)


class TransformationException(Exception):
    ...


@dataclass
class Pipeline(Generic[M]):
    transformations: Dict[Type[M], Dict[str, Transformation]] = field(
        default_factory=dict
    )
    lookup: Optional[Callable[[Type[M], Dict[str, Any]], M]] = None

    def create(self, model: Type[M], data: Any, *args, **kwargs) -> M:
        """Build a single instance of given model.
        
        :param model: model to build
        :param data: data to build instance from
        """
        return model(**self.build_kwargs(model, data, *args, **kwargs))  # type: ignore

    def create_multiple(
        self, model: Type[M], data: Sequence[Any], *args, **kwargs
    ) -> Iterator[M]:
        """Build multiple instance of given model.
        
        :param model: model to build
        :param data: data to build instances from
        """
        return (self.create(model, datapoint, *args, **kwargs) for datapoint in data)

    def get_or_create(
        self,
        model: Type[M],
        data: Any,
        *args,
        match_targets: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[M, bool]:
        """Get a single instance matching given match targets or create it.
        
        :param model: model to get or create the instance for
        :param data: data to get or create the instance from
        :param match_targets: fields to use for matching the data to the DB, default to None
        """
        lookup = kwargs.get("lookup") or getattr(self, "lookup")
        if lookup is None:
            raise NameError("Need to supply lookup to use `get or create` features.")

        if match_targets is None:
            lookup_kwargs = self.build_kwargs(model, data, *args, **kwargs)
        else:
            lookup_kwargs = {
                key: self.build_kwarg(model, key, data, *args, **kwargs)
                for key in match_targets
            }
        instance = lookup(model, lookup_kwargs)
        if not instance:
            if not match_targets:
                build_kwargs = self.build_kwargs(model, data, *args, **kwargs)
            return model(**build_kwargs), True  # type: ignore
        return instance, False

    def get_or_create_multiple(
        self,
        model: Type[M],
        data: Sequence[Any],
        *args,
        match_targets: Optional[List[str]] = None,
        **kwargs,
    ) -> Iterator[Tuple[M, bool]]:
        """Get a instances matching given match targets or create them.
        
        :param model: model to get or create the instances for
        :param data: data to get or create the instances from
        :param match_targets: fields to use for matching the data to the DB, default to None
        """
        return (
            self.get_or_create(
                model, datapoint, *args, match_targets=match_targets, **kwargs
            )
            for datapoint in data
        )

    def build_kwargs(
        self, model: Type[M], data: Any, *args, **kwargs
    ) -> Dict[str, Any]:
        """Build kwargs for an instance of given model.
        
        :param model: model to build kwargs for
        :param data: data to build kwargs from
        """
        return {
            name: self.build_kwarg(model, name, data, *args, **kwargs)
            for name, transformation in self.transformations[model].items()
        }

    def build_kwarg(
        self, model: Type[M], kwarg: str, data: Any, *args, **kwargs
    ) -> Dict[str, Any]:
        """Build a single kwarg for an instance of given model.
        
        :param model: model to build kwarg for
        :param kwarg: kwargs to build
        :param data: data to build kwarg from
        """
        try:
            return self.transformations[model][kwarg](self, data, *args, **kwargs)
        except TransformationException as e:
            e_type, transformation, *e_args = e.args
            raise e_type(
                f"failed @ {model.__name__}.{kwarg}: {transformation.__name__}: {e_args[0]}",
                *e_args[1:],
            )


class Transformation:
    def apply(self, pipeline: Pipeline, data: Any, *args, **kwargs) -> Any:
        raise NotImplementedError

    def __call__(self, pipeline: Pipeline, data: Any, *args, **kwargs) -> Any:
        try:
            return self.apply(pipeline, data, *args, **kwargs)
        except Exception as e:
            if isinstance(e, TransformationException):
                raise e
            raise TransformationException(type(e), type(self), *e.args)

    def __or__(self, other: Transformation) -> Chain:
        return Chain() | self | other


@dataclass
class Chain(Transformation):
    transformations: List[Transformation] = field(init=False, default_factory=list)

    def apply(self, pipeline: Pipeline, data: Any, *args, **kwargs) -> Any:
        val = data
        for transformation in self.transformations:
            val = transformation(pipeline, val, *args, **kwargs)
        return val

    def __or__(self, other: Transformation) -> Chain:
        self.transformations.append(other)
        return self
