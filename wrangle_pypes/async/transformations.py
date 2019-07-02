"""
This is a lot of duplication still looking for a cleaner way to
wrap the sync versions, which also enables proper co-op blocking.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    Type,
    Optional,
    Union,
    TypeVar,
    overload,
    Generic,
    Mapping,
    Sequence,
    Callable,
    Any,
    List,
    Iterable,
    Dict,
    Tuple,
)

from .pipeline import AsyncTransformation, AsyncPipeline

__all__ = (
    "Id",
    "Constant",
    "Custom",
    "Default",
    "Get",
    "Attr",
    "Filter",
    "Map",
    "ForEach",
    "Flatten",
    "Gather",
    "FoldInKeys",
    "FoldInValue",
    "GetKeys",
    "GetValues",
    "If",
    "Create",
    "CreateMultiple",
    "GetOrCreate",
    "GetOrCreateMultiple",
)

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class Id(AsyncTransformation, Generic[V]):
    async def apply(self, pipeline: AsyncPipeline, data: V, *args, **kwargs) -> V:
        return data


@dataclass
class Constant(AsyncTransformation, Generic[V]):
    value: V

    async def apply(self, pipeline: AsyncPipeline, data: Any, *args, **kwargs) -> V:
        return self.value


@dataclass
class Custom(AsyncTransformation, Generic[K, V]):
    func: Callable[[K], V]

    async def apply(self, pipeline: AsyncPipeline, data: K, *args, **kwargs) -> V:
        return self.func(data)  # type: ignore


@dataclass
class Default(AsyncTransformation, Generic[V, K]):
    value: V
    cond: Callable[[K], bool] = bool

    async def apply(self, pipeline: AsyncPipeline, data: K, *args, **kwargs) -> V:
        return data if self.cond(data) else self.value  # type: ignore


@dataclass
class Get(AsyncTransformation, Generic[K]):
    key: K
    default: Any = None

    @overload
    async def apply(self, pipeline, data: Mapping[K, V], *args, **kwargs) -> V:
        ...

    @overload
    async def apply(self, pipeline, data: Sequence[V], *args, **kwargs) -> V:
        ...

    async def apply(self, pipeline: AsyncPipeline, data, *args, **kwargs):
        try:
            return data[self.key]
        except (IndexError, KeyError):
            if self.default is not None:
                return self.default
            raise


@dataclass
class Attr(AsyncTransformation):
    attr: str

    async def apply(self, pipeline: AsyncPipeline, data: Any, *args, **kwargs) -> Any:
        return getattr(data, self.attr)


@dataclass
class Filter(AsyncTransformation, Generic[V]):
    func: Callable[[V], bool]

    async def apply(
        self, pipeline: AsyncPipeline, data: Iterable[V], *args, **kwargs
    ) -> List[V]:
        return [val for val in data if self.func(data)]  # type: ignore


@dataclass
class Map(AsyncTransformation, Generic[K, V]):
    func: Callable[[V], K]

    async def apply(
        self, pipeline: AsyncPipeline, data: Iterable[V], *args, **kwargs
    ) -> List[K]:
        return [self.func(val) for val in data]  # type: ignore


@dataclass
class ForEach(AsyncTransformation):
    transformation: AsyncTransformation

    async def apply(
        self, pipeline: AsyncPipeline, data: Sequence[Any], *args, **kwargs
    ) -> List:
        return [
            self.transformation(pipeline, datapoint, *args, **kwargs)
            for datapoint in data
        ]


@dataclass
class Flatten(AsyncTransformation):
    depth: int = 1

    async def apply(
        self, pipeline: AsyncPipeline, data: Sequence[Sequence], *args, **kwargs
    ) -> List:
        result = data
        for _ in range(self.depth):
            result = sum(result, [])
        return result  # type: ignore


@dataclass
class Gather(AsyncTransformation, Generic[K]):
    keys: Tuple[K]

    async def apply(
        self, pipeline: AsyncPipeline, data: Mapping[K, V], *args, **kwargs
    ) -> Dict[K, V]:
        return {key: data[key] for key in self.keys}


@dataclass
class FoldInKeys(AsyncTransformation):
    name: str

    async def apply(
        self, pipeline: AsyncPipeline, data: Mapping[Any, Mapping], *args, **kwargs
    ) -> List[Mapping]:
        return [{self.name: key, **datapoint} for key, datapoint in data.items()]


@dataclass
class FoldInValue(AsyncTransformation):
    key: str
    name: str

    async def apply(
        self, pipeline: AsyncPipeline, data: Mapping[Any, Any], *args, **kwargs
    ) -> Mapping:
        return {
            key: {self.name: data[self.key], **datapoint}
            for key, datapoint in data.items()
            if key != self.key
        }


@dataclass
class GetKeys(AsyncTransformation):
    async def apply(
        self, pipeline: AsyncPipeline, data: Mapping[K, V], *args, **kwargs
    ) -> List[K]:
        return list(data.keys())


@dataclass
class GetValues(AsyncTransformation):
    async def apply(
        self, pipeline: AsyncPipeline, data: Mapping[K, V], *args, **kwargs
    ) -> List[V]:
        return list(data.values())


@dataclass
class If(AsyncTransformation, Generic[V]):
    cond: Callable[[V], bool]
    then: AsyncTransformation
    else_: Optional[AsyncTransformation] = None

    async def apply(self, pipeline: AsyncPipeline, data: V, *args, **kwargs) -> Any:
        if self.cond(data):  # type: ignore
            return self.then(pipeline, data, *args, **kwargs)
        elif self.else_ is not None:
            return self.else_(pipeline, data, *args, **kwargs)
        else:
            return None


@dataclass
class Create(AsyncTransformation, Generic[V]):
    model: Type[V]

    async def apply(self, pipeline: AsyncPipeline, data: Any, *args, **kwargs) -> V:
        return await pipeline.create(self.model, data, *args, **kwargs)


@dataclass
class CreateMultiple(AsyncTransformation, Generic[V]):
    model: Type[V]

    async def apply(
        self, pipeline: AsyncPipeline, data: Sequence[Any], *args, **kwargs
    ) -> List[V]:
        return list(await pipeline.create_multiple(self.model, data, *args, **kwargs))


@dataclass
class GetOrCreate(AsyncTransformation, Generic[V]):
    model: Type[V]
    match_targets: Optional[List[str]] = None

    async def apply(
        self, pipeline: AsyncPipeline, data: Any, *args, **kwargs
    ) -> Tuple[V, bool]:
        return await pipeline.get_or_create(
            self.model, data, self.match_targets, *args, **kwargs
        )


@dataclass
class GetOrCreateMultiple(AsyncTransformation, Generic[V]):
    model: Type[V]
    match_targets: Optional[List[str]] = None

    async def apply(
        self, pipeline: AsyncPipeline, data: Sequence[Any], *args, **kwargs
    ) -> List[Tuple[V, bool]]:
        return list(
            await pipeline.get_or_create_multiple(
                self.model, data, self.match_targets, *args, **kwargs
            )
        )

