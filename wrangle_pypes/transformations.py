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

from .pipeline import Transformation, Pipeline

__all__ = (
    "Id",
    "Constant",
    "Cast",
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
class Id(Transformation, Generic[V]):
    def apply(self, pipeline: Pipeline, data: V, *args, **kwargs) -> V:
        return data


@dataclass
class Constant(Transformation, Generic[V]):
    value: V

    def apply(self, pipeline: Pipeline, data: Any, *args, **kwargs) -> V:
        return self.value


@dataclass
class Cast(Transformation, Generic[K, V]):
    func: Callable[[K], V]

    def apply(self, pipeline: Pipeline, data: K, *args, **kwargs) -> V:
        return self.func(data)  # type: ignore


@dataclass
class Custom(Transformation, Generic[K, V]):
    func: Callable[[Pipeline, K, Tuple, Dict[str, Any]], V]

    def apply(self, pipeline: Pipeline, data: K, *args, **kwargs) -> V:
        return self.func(pipeline, data, args, kwargs)  # type: ignore


@dataclass
class Default(Transformation, Generic[V, K]):
    value: V
    cond: Callable[[K], bool] = bool

    def apply(self, pipeline: Pipeline, data: K, *args, **kwargs) -> V:
        return data if self.cond(data) else self.value  # type: ignore


@dataclass
class Get(Transformation, Generic[K]):
    key: K
    default: Any = None

    @overload
    def apply(self, pipeline, data: Mapping[K, V], *args, **kwargs) -> V:
        ...

    @overload
    def apply(self, pipeline, data: Sequence[V], *args, **kwargs) -> V:
        ...

    def apply(self, pipeline: Pipeline, data, *args, **kwargs):
        try:
            return data[self.key]
        except (IndexError, KeyError):
            if self.default is not None:
                return self.default
            raise


@dataclass
class Attr(Transformation):
    attr: str

    def apply(self, pipeline: Pipeline, data: Any, *args, **kwargs) -> Any:
        return getattr(data, self.attr)


@dataclass
class Filter(Transformation, Generic[V]):
    func: Callable[[V], bool]

    def apply(self, pipeline: Pipeline, data: Iterable[V], *args, **kwargs) -> List[V]:
        return [val for val in data if self.func(data)]  # type: ignore


@dataclass
class Map(Transformation, Generic[K, V]):
    func: Callable[[V], K]

    def apply(self, pipeline: Pipeline, data: Iterable[V], *args, **kwargs) -> List[K]:
        return [self.func(val) for val in data]  # type: ignore


@dataclass
class ForEach(Transformation):
    transformation: Transformation

    def apply(self, pipeline: Pipeline, data: Sequence[Any], *args, **kwargs) -> List:
        return [
            self.transformation(pipeline, datapoint, *args, **kwargs)
            for datapoint in data
        ]


@dataclass
class Flatten(Transformation):
    depth: int = 1

    def apply(
        self, pipeline: Pipeline, data: Sequence[Sequence], *args, **kwargs
    ) -> List:
        result = data
        for _ in range(self.depth):
            result = sum(result, [])
        return result  # type: ignore


@dataclass
class Gather(Transformation, Generic[K]):
    keys: Tuple[K]

    def apply(
        self, pipeline: Pipeline, data: Mapping[K, V], *args, **kwargs
    ) -> Dict[K, V]:
        return {key: data[key] for key in self.keys}


@dataclass
class FoldInKeys(Transformation):
    name: str

    def apply(
        self, pipeline: Pipeline, data: Mapping[Any, Mapping], *args, **kwargs
    ) -> List[Mapping]:
        return [{self.name: key, **datapoint} for key, datapoint in data.items()]


@dataclass
class FoldInValue(Transformation):
    key: str
    name: str

    def apply(
        self, pipeline: Pipeline, data: Mapping[Any, Any], *args, **kwargs
    ) -> Mapping:
        return {
            key: {self.name: data[self.key], **datapoint}
            for key, datapoint in data.items()
            if key != self.key
        }


@dataclass
class GetKeys(Transformation):
    def apply(
        self, pipeline: Pipeline, data: Mapping[K, V], *args, **kwargs
    ) -> List[K]:
        return list(data.keys())


@dataclass
class GetValues(Transformation):
    def apply(
        self, pipeline: Pipeline, data: Mapping[K, V], *args, **kwargs
    ) -> List[V]:
        return list(data.values())


@dataclass
class If(Transformation, Generic[V]):
    cond: Callable[[V], bool]
    then: Transformation
    else_: Optional[Transformation] = None

    def apply(self, pipeline: Pipeline, data: V, *args, **kwargs) -> Any:
        if self.cond(data):  # type: ignore
            return self.then(pipeline, data, *args, **kwargs)
        elif self.else_ is not None:
            return self.else_(pipeline, data, *args, **kwargs)
        else:
            return None


@dataclass
class Create(Transformation, Generic[V]):
    model: Type[V]

    def apply(self, pipeline: Pipeline, data: Any, *args, **kwargs) -> V:
        return pipeline.create(self.model, data, *args, **kwargs)


@dataclass
class CreateMultiple(Transformation, Generic[V]):
    model: Type[V]

    def apply(
        self, pipeline: Pipeline, data: Sequence[Any], *args, **kwargs
    ) -> List[V]:
        return list(pipeline.create_multiple(self.model, data, *args, **kwargs))


@dataclass
class GetOrCreate(Transformation, Generic[V]):
    model: Type[V]
    match_targets: Optional[List[str]] = None

    def apply(self, pipeline: Pipeline, data: Any, *args, **kwargs) -> Tuple[V, bool]:
        return pipeline.get_or_create(
            self.model, data, self.match_targets, *args, **kwargs
        )


@dataclass
class GetOrCreateMultiple(Transformation, Generic[V]):
    model: Type[V]
    match_targets: Optional[List[str]] = None

    def apply(
        self, pipeline: Pipeline, data: Sequence[Any], *args, **kwargs
    ) -> List[Tuple[V, bool]]:
        return list(
            pipeline.get_or_create_multiple(
                self.model, data, self.match_targets, *args, **kwargs
            )
        )

