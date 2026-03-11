from huggingface_hub import snapshot_download
folder = snapshot_download(
                "HuggingFaceFW/fineweb", 
                repo_type="dataset",
                local_dir="/data/sknigam/fineweb_10bt",
                local_dir_use_symlinks=False,
                # replace "data/CC-MAIN-2023-50/*" with "sample/100BT/*" to use the 100BT sample
                allow_patterns="sample/10BT/*",
                max_workers=5
                )
