import hashlib
import shutil
import os
import tempfile
import zipfile
import logging
import sys
import json
from pathlib import Path
import numpy as np
from ..types import Dataset, Cameras, CameraModel
from ._common import DatasetNotFoundError


SCANNERF_SCENES = { "airplane1", "airplane2", "brontosaurus", "bulldozer1", "bulldozer2", "cheetah", "dump_truck1", "dump_truck2", "elephant", "excavator", "forklift", "giraffe", "helicopter1", "helicopter2", "lego", "lion", "plant1", "plant2", "plant3", "plant4", "plant5", "plant6", "plant7", "plant8", "plant9", "roadroller", "shark", "spinosaurus", "stegosaurus", "tiger", "tractor", "trex", "triceratops", "truck", "zebra"}
SCANNERF_SPLITS = {"train", "test", "val"}


def load_scannerf_dataset(path: Path, split: str, **kwargs):
    assert isinstance(path, (Path, str)), "path must be a pathlib.Path or str"
    path = Path(path)

    scene = path.name
    if scene not in SCANNERF_SCENES:
        raise DatasetNotFoundError(f"Scene {scene} not found in Scannerf dataset. Supported scenes: {SCANNERF_SCENES}.")
    for dsplit in SCANNERF_SPLITS:
        dsuffix = "0" if dsplit == "train" else "0"
        if not (path / f"{dsplit}_{dsuffix}.json").exists():
            raise DatasetNotFoundError(f"Path {path} does not contain a Scannerf dataset. Missing file: {path / f'{dsplit}_{dsuffix}.json'}")

    assert split in SCANNERF_SPLITS, "split must be one of 'train', 'test' or 'val'"

    suffix = "0" if split == "train" else "0"
    with (path / f"{split}_{suffix}.json").open("r", encoding="utf8") as fp:
        meta = json.load(fp)

    cams = []
    image_paths = []
    for _, frame in enumerate(meta["frames"]):
        fprefix = path / frame["file_path"]
        image_paths.append(str(fprefix) + ".png")
        cams.append(np.array(frame["transform_matrix"], dtype=np.float32))

    w = int(meta["w"])
    h = int(meta["h"])
    image_sizes = np.array([w, h], dtype=np.int32)[None].repeat(len(cams), axis=0)
    nears_fars = np.array([0.05, 10.0], dtype=np.float32)[None].repeat(len(cams), axis=0)
    fx = float(meta["fl_x"])
    fy = float(meta["fl_y"])
    cx = float(meta["cx"])
    cy = float(meta["cy"])
    intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)[None].repeat(len(cams), axis=0)
    distortion_params = np.repeat(
        np.array(
            [
                float(meta["k1"]) if "k1" in meta else 0.0,
                float(meta["k2"]) if "k2" in meta else 0.0,
                float(meta["p2"]) if "p1" in meta else 0.0,
                float(meta["p1"]) if "p2" in meta else 0.0,
                float(meta["k3"]) if "k3" in meta else 0.0,
                float(meta["k4"]) if "k4" in meta else 0.0,
            ]
        )[None, :],
        len(cams),
        0,
    )
    return Dataset(
        cameras=Cameras(
            poses=np.stack(cams)[:, :3, :4],
            normalized_intrinsics=intrinsics / image_sizes[..., :1],
            camera_types=np.full(len(cams), CameraModel.OPENCV.value, dtype=np.int32),
            distortion_parameters=distortion_params,
            image_sizes=image_sizes,
            nears_fars=nears_fars,
        ),
        file_paths_root=path,
        file_paths=image_paths,
        sampling_mask_paths=None,
        color_space="srgb",
        metadata={
            "name": "scannerf",
            "scene": scene,
            "background_color": np.array([255, 255, 255], dtype=np.uint8),
        },
    )

def grab_file_id(zip_url: str) -> str:
    """Get the file id from the google drive zip url."""
    s = zip_url.split("/d/")[1]
    return s.split("/")[0]

scannerf_test_file_ids = {
    "airplane1": grab_file_id("https://drive.google.com/file/d/1z_5c4ssMlkvKIlKHkp0FRTr-FkSxkZnD/view"),
    "airplane2": grab_file_id("https://drive.google.com/file/d/1o3h4Lg07A1eK27TBHP156KVjDIk9FFNe/view"),
    "brontosaurus": grab_file_id("https://drive.google.com/file/d/1cSMVvdfSebUHmi-ygPziXVpQ8DpAijuH/view"),
    "bulldozer1": grab_file_id("https://drive.google.com/file/d/1o-floxlYGesAOGasMYf7PWNmv-FO6Vhv/view"),
    "bulldozer2": grab_file_id("https://drive.google.com/file/d/1lXQrps4WMnakSL58PiECmSmTft5LkizX/view"),
    "cheetah": grab_file_id("https://drive.google.com/file/d/1zT8ziBne_3S-fTh8e9cg-SV_qIig73OX/view"),
    "dump_truck1": grab_file_id("https://drive.google.com/file/d/10ZD8GEbwYLQNsiOCJnVi2Nk5G1555oj5/view"),
    "dump_truck2": grab_file_id("https://drive.google.com/file/d/1-bRNXD4mcFQuVvRwonwssaPiBS9XAYhl/view"),
    "elephant": grab_file_id("https://drive.google.com/file/d/1aJKbHhzPxs8Zxv2WSV5s7-Cz9wIxfsj6/view"),
    "excavator": grab_file_id("https://drive.google.com/file/d/1N-rQ0xE093nXFSvOa5leoCRelRFC6hZH/view"),
    "forklift": grab_file_id("https://drive.google.com/file/d/1qV2EPWj5yZ9XGfaq_5CMHaaOBsOsfhNH/view"),
    "giraffe": grab_file_id("https://drive.google.com/file/d/1snTuS6la5rHZhZOnsbI9jA0mVY53U8z-/view"),
    "helicopter1": grab_file_id("https://drive.google.com/file/d/1iiFYtG95Ai7xw9qS62m4dWas_Dvs4eqp/view"),
    "helicopter2": grab_file_id("https://drive.google.com/file/d/10S1EJB7_9P_o80FNI9q-k4aFDEbf6i3i/view"),
    "lego": grab_file_id("https://drive.google.com/file/d/1hckePa0mZqVtuUjb_-DX6DmjsudVEroE/view"),
    "lion": grab_file_id("https://drive.google.com/file/d/1FblWa1DxBR2kzntEmABLxO9Ue6nD0hGi/view"),
    "plant1": grab_file_id("https://drive.google.com/file/d/11EwJFhwCCUiIsqA8OQI_cPCIu44CogIT/view"),
    "plant2": grab_file_id("https://drive.google.com/file/d/15s-cHMq4VLPC2Ubkejjvw-dx-94NBdBu/view"),
    "plant3": grab_file_id("https://drive.google.com/file/d/1ilDP4AWSNRFbDIb6aMddS7QLOaI1btta/view"),
    "plant4": grab_file_id("https://drive.google.com/file/d/1a-m4U1wszPpQquFDC6S9ZVVacpieJ3o4/view"),
    "plant5": grab_file_id("https://drive.google.com/file/d/1JVPV7knFhv5jXcxW85iQPaoKPZsX-zWR/view"),
    "plant6": grab_file_id("https://drive.google.com/file/d/1TYXy7DHTwCK4FceHGDeIzjBbduRcKn9i/view"),
    "plant7": grab_file_id("https://drive.google.com/file/d/1d2qtTJjA0P7-joSNRDuALgwEuszljHlw/view"),
    "plant8": grab_file_id("https://drive.google.com/file/d/1yY4hNiWp7CPrO1aW-hfIn42CT02547tV/view"),
    "plant9": grab_file_id("https://drive.google.com/file/d/1az-52bkrJO3--IFhLPBFq6-rYxHBnWVh/view"),
    "roadroller": grab_file_id("https://drive.google.com/file/d/1j5roxbVA9zZuQZ9HwyTCpL5CnxJW_5yx/view"),
    "shark": grab_file_id("https://drive.google.com/file/d/1olmQqdbr1R9KW9yqK7T-Jp8rgh7GC6lq/view"),
    "spinosaurus": grab_file_id("https://drive.google.com/file/d/1DrpoHm0SnMdSvzwzhcFOTR7jk7-Etc9P/view"),
    "stegosaurus": grab_file_id("https://drive.google.com/file/d/1HqcIbkrom7n_Cq9gYgQIdTz-GX_U6C4f/view"),
    "tiger": grab_file_id("https://drive.google.com/file/d/1P2v82M9tZMYyJnfGHXexZ9c1FBOzd9z9/view"),
    "tractor": grab_file_id("https://drive.google.com/file/d/1dX0sJh-oROhwrTljC-4K6C6pNLMwWWI2/view"),
    "trex": grab_file_id("https://drive.google.com/file/d/1_kFMVqzw_J0htnBZIOh83FDzbE9_5LEq/view"),
    "triceratops": grab_file_id("https://drive.google.com/file/d/188cLLcRDcPMDB2vTkbd7cafHZiJ-DS3s/view"),
    "truck": grab_file_id("https://drive.google.com/file/d/1BVsG0oJaRqaIzKJYhzbYvq9CnV0gru_o/view"),
    "zebra": grab_file_id("https://drive.google.com/file/d/1q78uMLeg50yiBbcWK4MNPSEyCUh6yAFi/view"),
}
scannerf_file_ids = {
    "airplane1": grab_file_id("https://drive.google.com/file/d/10SHrYsbvKN7hb7bs8vEOoFi8gNOAdDOQ/view"),
    "airplane2": grab_file_id("https://drive.google.com/file/d/1XSQ0AVXbrVuaML_dePmo0_8FkdKfuyhl/view"),
    "brontosaurus": grab_file_id("https://drive.google.com/file/d/18GG581txZs65Jd_ChMJwid41UBguiF2H/view"),
    "bulldozer1": grab_file_id("https://drive.google.com/file/d/1ffpgj3oScKPBIxlhdqmrTFmN8dfkX2AB/view"),
    "bulldozer2": grab_file_id("https://drive.google.com/file/d/1knLhC4urgugkZjbIt8BC6zh5uB12B-fd/view"),
    "cheetah": grab_file_id("https://drive.google.com/file/d/1CjzOOc-U8lxOxv65ipEqbwCEpmG-sJRT/view"),
    "dump_truck1": grab_file_id("https://drive.google.com/file/d/1r1jXZU429Q4YlMX6C2wc1g-mQf-MkqBk/view"),
    "dump_truck2": grab_file_id("https://drive.google.com/file/d/1g8isRpCSv9LOTywaCIFc_7_THcsdCd5j/view"),
    "elephant": grab_file_id("https://drive.google.com/file/d/1Rc4aA-uINEYQHG5GRR5IWwl8Tb6MtPE8/view"),
    "excavator": grab_file_id("https://drive.google.com/file/d/17PnQzF0j4z1ar5ElUCdXUkFOoTL8Qg9Q/view"),
    "forklift": grab_file_id("https://drive.google.com/file/d/1hVFUK2DIyh6isUXSuhiVRyoQi3Kfh_j-/view"),
    "giraffe": grab_file_id("https://drive.google.com/file/d/1ARHk1v5dYw866e_5mqIqlWwplG4GRY9t/view"),
    "helicopter1": grab_file_id("https://drive.google.com/file/d/1VmRepcD8fBng2PPJBH-fupXxGWsCzNKB/view"),
    "helicopter2": grab_file_id("https://drive.google.com/file/d/1sg3uwkYcp4ejVERZozhQGar9VasRZLZc/view"),
    "lego": grab_file_id("https://drive.google.com/file/d/1NGhl9QHorjra3VMMvhfNL_3YBnDWjepi/view"),
    "lion": grab_file_id("https://drive.google.com/file/d/1pQn39oXwyFY5YGIiAxyWydRzenQoE3iN/view"),
    "plant1": grab_file_id("https://drive.google.com/file/d/1u8dHH7NSq5Qc3mtQ5UGd4oHZM-YNMNKF/view"),
    "plant2": grab_file_id("https://drive.google.com/file/d/1l1ywWFEIV0xpIYQdCzX_Crj2SK7qDCMp/view"),
    "plant3": grab_file_id("https://drive.google.com/file/d/1eLKoDqNZqcDB41-_-g2SpuAlQvxPce9J/view"),
    "plant4": grab_file_id("https://drive.google.com/file/d/1PmHpqqf-ijhdsYw0pZOmPjKmRMHxl9yU/view"),
    "plant5": grab_file_id("https://drive.google.com/file/d/1RRWP_PcV7rSFWMZgmm3cjzdZaoEkhiGg/view"),
    "plant6": grab_file_id("https://drive.google.com/file/d/1Ii5VqnaRJ59VWQFyDyXg3MKZ8aYiL0_T/view"),
    "plant7": grab_file_id("https://drive.google.com/file/d/1dAyyCVgiebEg3OEs_KNNbfmaKgA0l21_/view"),
    "plant8": grab_file_id("https://drive.google.com/file/d/1xE-xdw6DUkkYSS1XP7SVCTpMeKJdWIA_/view"),
    "plant9": grab_file_id("https://drive.google.com/file/d/1zvwWpgucYKqmU04oyXmlsGChKL1Ifuga/view"),
    "roadroller": grab_file_id("https://drive.google.com/file/d/1mvubO4b2323Y5E0ktlkYHRgIqloJSSTP/view"),
    "shark": grab_file_id("https://drive.google.com/file/d/1Dy1rKLO0VKfXw0MVabKGE-bDQ0QZOWFa/view"),
    "spinosaurus": grab_file_id("https://drive.google.com/file/d/1_TjRrPrCbW8esg0XnyBRqSHtDTk0wRS8/view"),
    "stegosaurus": grab_file_id("https://drive.google.com/file/d/1NtIn5eAmKYis1XUTnaIXwJs2PgRIuyKs/view"),
    "tiger": grab_file_id("https://drive.google.com/file/d/1baM8aLJEr8BQuvftCK6tb5Aytj4UpIJR/view"),
    "tractor": grab_file_id("https://drive.google.com/file/d/1P1SiuKWG0mFTCiEtQECr3Admud5pHLQY/view"),
    "trex": grab_file_id("https://drive.google.com/file/d/1OzMcvmGY6CFXG8SxqLORQNT40E8Exdpm/view"),
    "triceratops": grab_file_id("https://drive.google.com/file/d/1Ekqh2K3SsInd-1r0r7eZvOSWkuq9kZZC/view"),
    "truck": grab_file_id("https://drive.google.com/file/d/13CBvLCK5ZNqKqvayxpTkEYDAcqZRwYIh/view"),
    "zebra": grab_file_id("https://drive.google.com/file/d/1p751v_dWYRvgjYWaUPlF8RbS5JHVn4AZ/view"),
}

import os
import shutil
import subprocess
import logging
import sys
from pathlib import Path
import tarfile

def download_capture_name(output: Path, file_id_or_tar_url, is_test=False):
    """Download specific captures from a given dataset and capture name, handling .tar files."""
    target_path = str(output)
    download_path = Path(f"{target_path}_private.tar" if is_test else f"{target_path}.tar")
    tmp_path = target_path + ".tmp"
    shutil.rmtree(tmp_path, ignore_errors=True)
    os.makedirs(tmp_path, exist_ok=True)
    try:
        os.remove(download_path)
    except OSError:
        pass
    if file_id_or_tar_url.endswith(".tar"):
        url = file_id_or_tar_url  # tar url
        subprocess.check_call(f"wget {url} -O {download_path}")
    else:
        try:
            import gdown
        except ImportError:
            logging.fatal("Please install gdown: pip install gdown")
            sys.exit(2)
        url = f"https://drive.google.com/uc?id={file_id_or_tar_url}"  # file id
        try:
            os.remove(download_path)
        except OSError:
            pass
        gdown.download(url, output=str(download_path))
    with tarfile.open(download_path, "r:*") as tar_ref:
        tar_ref.extractall(tmp_path)
    inner_folders = os.listdir(tmp_path)
    assert len(inner_folders) == 1, "There is more than one folder inside this tar file."
    folder = os.path.join(tmp_path, inner_folders[0])
    #shutil.rmtree(target_path, ignore_errors=True)
    
    for item in os.listdir(folder):
        s = os.path.join(folder, item)
        d = os.path.join(target_path, item)
        if os.path.isdir(s):
            shutil.move(s, d)
        else:
            shutil.copy2(s, d)
    shutil.rmtree(tmp_path)
    os.remove(download_path)


def download_scannerf_dataset(path: str, output: Path):
    output = Path(output)
    if not path.startswith("scannerf/") and path != "scannerf":
        raise DatasetNotFoundError("Dataset path must be equal to 'scannerf' or must start with 'scannerf/'.")
    if path == "scannerf":
        for x in scannerf_file_ids:
            download_scannerf_dataset(f"scannerf/{x}", output / x)
        return
    capture_name = path[len("scannerf/") :]
    download_capture_name(output, scannerf_test_file_ids[capture_name], is_test=True)
    download_capture_name(output, scannerf_file_ids[capture_name])
    logging.info(f"Downloaded {path} to {output}")