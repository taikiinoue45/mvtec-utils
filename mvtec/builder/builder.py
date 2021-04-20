from importlib import import_module
from typing import Any, List

from omegaconf import DictConfig, ListConfig


class Builder:
    def build_list_cfg(self, cfg: ListConfig) -> List[Any]:

        return [self._get_attr(c.name)(**c.get("args", {})) for c in cfg]

    def build_dict_cfg(self, cfg: DictConfig, **kargs: Any) -> Any:

        return self._get_attr(cfg.name)(**cfg.get("args", {}), **kargs)

    def _get_attr(self, name: str) -> Any:

        module_path, attr_name = name.split(" - ")
        module = import_module(module_path)
        return getattr(module, attr_name)
