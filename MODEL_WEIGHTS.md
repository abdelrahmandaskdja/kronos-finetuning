# Public model weights

The complete weight archive is hosted at:

**https://huggingface.co/abdelrahman964/kronos-finetuning**

The Hugging Face repository contains all 11 saved weight files from the
supplied project archive. The three selected checkpoints are:

| Horizon | Hugging Face path | Bytes | SHA-256 |
|---|---|---:|---|
| 5 minutes | `models/5min-native-direction-head/best_direction_model.pt` | 412,091,429 | `a5eb997e10847cbca5d6c4904112e62eef495933a5ca8f4adbc192b37bd7283b` |
| 1 hour | `models/1h-pruned50-direction-head/best_direction_model.pt` | 409,312,675 | `037bb9e88179069f7ee36d40aeaf1134f92e5c75311590f59a67be2cd07c07f8` |
| 1 day | `models/1d-last6-path-adapter/model.safetensors` | 412,036,408 | `62835a901a36720d2a1ee2fa540f15430806d66cc9e5a08140a92f143ef48c82` |

Supporting and experimental weights:

| Hugging Face path | Bytes | SHA-256 |
|---|---:|---|
| `models/5min-kronos-base/model.safetensors` | 409,264,008 | `caf9eef16a6bae2cb9519b9586086aa0423db9bf28179a2d3b3f2b05d01ab8bb` |
| `models/5min-tokenizer/model.safetensors` | 15,842,368 | `8bfc46fb5f2312f31e8cb7424772d0826c54e5c951a5dad7c61c98c407b4cef8` |
| `models/1m-trainall-2025h1/base/model.safetensors` | 412,036,408 | `628e255c15594295c23b26a62adeea2d32cc6e944b61b36aeb94402fb9d450f6` |
| `models/1m-trainall-2025h1/tokenizer/model.safetensors` | 15,842,368 | `5641cc0d1a6357f448fa34817abc64a2a91301ea5914ef00910890956cf56a59` |
| `models/1m-trainall-2025h1/rollout-checkpoint/model.safetensors` | 412,036,408 | `ca7923c4e6ac80c2c15fecfb02bf547e4ce0d5d8e474948b03d4d764900513ae` |
| `models/1m-retrain-2017-2024/base/model.safetensors` | 412,036,408 | `db59dc76b2b1a77a5a6433459ccf7bac85c28769775f851b037d84a1e6782942` |
| `models/1m-retrain-2017-2024/tokenizer/model.safetensors` | 15,842,368 | `9792ef7e627c5908f1e1bf92df2b1a4e3352b8a50037b2fa369248c3a2810c77` |
| `models/1m-retrain-2017-2024/rollout-checkpoint/model.safetensors` | 412,036,408 | `db59dc76b2b1a77a5a6433459ccf7bac85c28769775f851b037d84a1e6782942` |

The two final `1m-retrain-2017-2024` files are byte-identical. Both paths are
published because both were present in the supplied archive. Hugging Face's
content-addressed storage can deduplicate the underlying blob.

The machine-readable `weights_manifest.json` in the Hugging Face repository
records the original archive paths and the public paths used above.

## Download

```python
from huggingface_hub import snapshot_download

model_root = snapshot_download(
    repo_id="abdelrahman964/kronos-finetuning",
    allow_patterns=["models/**", "weights_manifest.json"],
)
print(model_root)
```

The `.pt` checkpoints use the custom direction-head implementation in
`finetune_csv/kronos_direction_model.py`. Use only checkpoints from trusted
sources and verify the SHA-256 values before loading them.
