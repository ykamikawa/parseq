from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

from strhub.data.module import SceneTextDataModule
from strhub.models.parseq.system import PARSeq


# カスタムデータモジュールを作成
class CustomDataModule(SceneTextDataModule):
    def test_dataloader(self):
        # val_dataloaderと同じデータセットを使用
        return self.val_dataloader()


@hydra.main(config_path="configs", config_name="main", version_base="1.2")
def main(config: DictConfig):
    # 固定シード値を使用
    pl.seed_everything(42)

    # カスタムデータモジュールの作成
    dm = CustomDataModule(
        root_dir=config.data.root_dir,
        train_dir=config.data.train_dir,
        batch_size=config.data.batch_size,
        img_size=config.data.img_size,
        charset_train=config.data.charset_train,
        charset_test=config.data.charset_test,
        max_label_length=config.data.max_label_length,
        remove_whitespace=config.data.remove_whitespace,
        normalize_unicode=config.data.normalize_unicode,
        augment=config.data.augment,
        num_workers=config.data.num_workers,
    )

    # 学習済みモデルのロード
    print("Loading pretrained Parseq model...")
    model = PARSeq(**config.model)

    pretrained_state_dict = torch.hub.load(
        "baudm/parseq", "parseq", pretrained=True
    ).state_dict()

    # 選択的にパラメータを転送
    if pretrained_state_dict:
        current_model_dict = model.state_dict()
        new_state_dict = {}
        keys_to_ignore = []
        print("Checking shape compatibility...")
        for k, v in pretrained_state_dict.items():
            if k in current_model_dict and v.shape == current_model_dict[k].shape:
                new_state_dict[k] = v
            else:
                # 形状が異なる、または現在のモデルに存在しないキーは無視
                if k in current_model_dict:
                    print(
                        f"  Ignoring '{k}': shape mismatch. Pretrained: {v.shape}, Current: {current_model_dict[k].shape}"
                    )
                else:
                    print(f"  Ignoring '{k}': key not found in current model.")
                keys_to_ignore.append(k)

        if not keys_to_ignore:
            print("  All weights are compatible.")

        # strict=False なので、new_state_dictに含まれないキーは現在のモデルの初期値が維持される
        load_result = model.load_state_dict(new_state_dict, strict=False)
        # load_result.missing_keys は new_state_dict になかった現在のモデルのキー
        # load_result.unexpected_keys は常に空のはず (strict=False なので)
        print(
            f"Weights loaded. Missing keys in loaded state_dict (expected): {len(load_result.missing_keys)}"
        )
        if keys_to_ignore:
            print(f"Ignored keys from pretrained_state_dict: {keys_to_ignore}")

    else:
        print("No weights transferred, using initialized model.")
        raise ValueError("No weights transferred, using initialized model.")

    # 出力ディレクトリを設定
    # Hydra の出力ディレクトリを使用
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_accuracy",
        dirpath=output_dir,
        filename="model_best",
        save_top_k=1,
        mode="max",
    )

    # モデルの訓練
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        val_check_interval=config.trainer.val_check_interval,
        max_epochs=config.trainer.max_epochs,
        gradient_clip_val=config.trainer.gradient_clip_val,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        logger=pl.loggers.TensorBoardLogger(
            save_dir=output_dir,
            name="",
            version="",
            log_graph=False,
        ),
        callbacks=[checkpoint_callback],
    )

    # train
    trainer.fit(model, datamodule=dm)

    # test
    trainer.test(model, dm)

    return trainer.callback_metrics


if __name__ == "__main__":
    main()
