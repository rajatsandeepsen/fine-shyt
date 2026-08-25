from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeVar, cast, overload

from pydantic import BaseModel

DumpMode = Literal["python", "json"]
ModelT = TypeVar("ModelT", bound=BaseModel)
KeyT = TypeVar("KeyT")


@overload
def pydantic_to_python(
    data: ModelT,
    *,
    mode: DumpMode = "python",
    by_alias: bool = False,
    exclude_none: bool = False,
) -> dict[str, Any]: ...


@overload
def pydantic_to_python(
    data: Sequence[ModelT],
    *,
    mode: DumpMode = "python",
    by_alias: bool = False,
    exclude_none: bool = False,
) -> list[dict[str, Any]]: ...


@overload
def pydantic_to_python(
    data: Mapping[KeyT, ModelT],
    *,
    mode: DumpMode = "python",
    by_alias: bool = False,
    exclude_none: bool = False,
) -> dict[KeyT, dict[str, Any]]: ...


def pydantic_to_python(
    data: BaseModel | Sequence[BaseModel] | Mapping[KeyT, BaseModel],
    *,
    mode: DumpMode = "python",
    by_alias: bool = False,
    exclude_none: bool = False,
):
    if isinstance(data, BaseModel):
        return data.model_dump(
            mode=mode,
            by_alias=by_alias,
            exclude_none=exclude_none,
        )

    if isinstance(data, Mapping):
        return {
            key: cast(value, BaseModel).model_dump(
                mode=mode,
                by_alias=by_alias,
                exclude_none=exclude_none,
            )
            for key, value in data.items()
        }

    if isinstance(data, Sequence) and not isinstance(
        data, (str, bytes, bytearray)
    ):
        return [
            item.model_dump(
                mode=mode,
                by_alias=by_alias,
                exclude_none=exclude_none,
            )
            for item in data
        ]

    raise TypeError(
        "data must be a BaseModel, a sequence of BaseModel, or a mapping of BaseModel"
    )
