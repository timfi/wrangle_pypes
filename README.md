# wrangle_pypes
_Easy, model based data wrangling._

<a href="https://mypy-lang.org/"><img alt="Checked with mypy" src="http://www.mypy-lang.org/static/mypy_badge.svg"></a>
<a href="https://github.com/python/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

## Origin
This project came about thanks to having implemented this concept in a prior project ([dfb_predict](https://github.com/tim-fi/dfb_predict)) and wanting to reuse it. In said previous project it came about, because we wanted to play around with data-oriented design and figured that if we break up the transformations required during data-wrangling/munging into a simple "_AST_" where each node represents a single transformation. And to further the idea we to some inspiration from one of the [Unix philosophy](https://en.wikipedia.org/wiki/Unix_philosophy) core tenants, i.e. a single transformation should do one simple thing only, and allow for composition to implement more complex behavior.

## Example
```python
from dataclasses import dataclass
import json

from wrangle_pypes import Pipeline
from wrangle_pypes.transformations import Get, Create, Cast

@dataclass
class Point:
    x: int
    y: int


@dataclass
class Square:
    A: Point
    B: Point


pipeline = Pipline({
    Square: {
        "A": Get("A") | Create(Point),
        "A": Get("B") | Create(Point),
    },
    Point: {
        "x": Get("x") | Cast(int),
        "y": Get("y") | Cast(int),
    }
})

data = """
[
    {
        "A": {"x": 0, "y", 0},
        "B": {"x": 1, "y", 1},
    }, {
        "A": {"x": 10, "y", 10},
        "B": {"x": 11, "y", 11},
    }
]
"""

squares = list(pipeline.create_multiple(json.loads(data)))
```
