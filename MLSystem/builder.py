# system/builder.py
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

# 環境変数を設定し、sys.pathに必要なパスを追加


from MLsystem.registry import Registry
from MLsystem.hashing import compute_combined_hash
from MLsystem.inspector import find_config_class
from MLsystem.utils.env_manager import EnvManager

@dataclass
class ExperimentContext:
    """実験構築の結果を保持するコンテキスト"""
    hash_id: str
    diff_payload: Dict[str, Any]
    model: pl.LightningModule
    datamodule: pl.LightningDataModule
    all_params: Dict[str, Any]
    
    # 構築に使ったクラス定義（ログ保存用）
    model_cls: Any
    adapter_cls: Any
    data_cls: Any

class ExperimentBuilder:
    """
    Hydraの設定(cfg)を受け取り、実験に必要なオブジェクト(Model, DataModule)を
    適切な依存関係と設定マージ順序で構築するクラス。
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.registry = Registry()

    def build(self) -> ExperimentContext:
        # 1. モジュールとクラス定義の読み込み
        modules = self._load_definitions()
        
        # 2. ユーザー設定の展開
        user_params = self._extract_user_params()
        
        # 3. ハッシュ計算
        hash_id, diff_payload = self._compute_hash(modules, user_params)
        
        # 4. パラメータの統合 (Common < Dataset < Adapter < Model)
        all_params = self._merge_all_params(user_params)
        
        # 5. DataModuleの構築
        input_transform = modules["adapter_mod"].get_input_transform(user_params["adapter"])
        
        datamodule = modules["DataModuleClass"](
            adapter_transform=input_transform, 
            **all_params
        )
        
        # 6. Setup実行とメタ情報取得
        # Modelの初期化引数を決定するために必要
        datamodule.prepare_data()
        datamodule.setup(stage="fit")
        
        data_meta = {k: getattr(datamodule, k) for k in dir(datamodule) if not k.startswith('_')}
        
        # 7. Modelの構築
        # Adapter経由でDataModuleの情報をModel引数に変換
        model_init_args = modules["adapter_mod"].get_model_init_args(data_meta, user_params["adapter"])
        
        # Model引数の結合 (all_params + model_init_args)
        # model_init_args(動的) がユーザー設定より優先されるべきか、あるいは逆かは設計次第だが、
        # ここでは「統合パラメータ」に「動的パラメータ」を上書きする形でModelに渡す
        final_model_kwargs = {**all_params, **model_init_args}
        
        model = modules["ModelClass"](**final_model_kwargs)
        
        return ExperimentContext(
            hash_id=hash_id,
            diff_payload=diff_payload,
            model=model,
            datamodule=datamodule,
            all_params=all_params,
            model_cls=modules["model_conf_cls"],
            adapter_cls=modules["adapter_conf_cls"],
            data_cls=modules["data_conf_cls"]
        )

    def _load_definitions(self):
        cfg = self.cfg
        try:
            return {
                "ModelClass": self.registry.get_main_class("models", cfg.model),
                "DataModuleClass": self.registry.get_main_class("datasets", cfg.dataset),
                "adapter_mod": self.registry.get_adapter_module(cfg.model, cfg.adapter),
                
                "model_conf_cls": self.registry.get_config_class("models", cfg.model),
                "adapter_conf_cls": self.registry.get_config_class("models", cfg.model, cfg.adapter),
                "data_conf_cls": self.registry.get_config_class("datasets", cfg.dataset),
                "common_conf_cls": find_config_class(EnvManager().get_common_config_module())
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load definition modules: {e}")

    def _extract_user_params(self):
        cfg = self.cfg
        # HydraのDictConfigを通常のdictに変換
        return {
            "common": OmegaConf.to_container(cfg.common, resolve=True) if "common" in cfg else {},
            "model": OmegaConf.to_container(cfg.model_params, resolve=True) if "model_params" in cfg else {},
            "adapter": OmegaConf.to_container(cfg.adapter_params, resolve=True) if "adapter_params" in cfg else {},
            "data": OmegaConf.to_container(cfg.data_params, resolve=True) if "data_params" in cfg else {},
            # 名前情報も追加
            "model_name": cfg.model,
            "adapter_name": cfg.adapter,
            "dataset_name": cfg.dataset
        }

    def _compute_hash(self, modules, user_params):
        # パラメータdictに_nameを注入してハッシュ計算
        p_common = user_params["common"]
        p_model = {**user_params["model"], "_name": user_params["model_name"]}
        p_adapter = {**user_params["adapter"], "_name": user_params["adapter_name"]}
        p_data = {**user_params["data"], "_name": user_params["dataset_name"]}

        return compute_combined_hash(
            modules["common_conf_cls"], p_common,
            modules["model_conf_cls"], p_model,
            modules["adapter_conf_cls"], p_adapter,
            modules["data_conf_cls"], p_data
        )

    def _merge_all_params(self, user_params):
        """
        全パラメータを統合する (Common < Dataset < Adapter < Model の順で上書き)
        """
        all_params = {}
        all_params.update(user_params["common"])
        all_params.update(user_params["data"])
        all_params.update(user_params["adapter"])
        all_params.update(user_params["model"])
        return all_params