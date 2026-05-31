$ErrorActionPreference = "Stop"

Write-Host "1/3 Static Python compile checks"
python -m py_compile analyze_pkl.py data_loader.py visualize_metrics.py
python -m py_compile resnet\model.py resnet\train.py resnet\predict.py resnet\batch_predict.py resnet\load_weights.py resnet\evaluate.py
python -m py_compile grad_cam\main_cnn.py grad_cam\main_vit.py grad_cam\main_swin.py grad_cam\utils.py grad_cam\vit_model.py grad_cam\swin_model.py grad_cam\resnet_grad_cam.py

Write-Host ""
Write-Host "2/3 Dataset summary artifacts"
if (Test-Path dataset) {
    python scripts\summarize_dataset.py --dataset-dir dataset --output-dir reports
} else {
    Write-Host "dataset directory not found; run python data_loader.py after downloading LSWMD.pkl"
}

Write-Host ""
Write-Host "3/3 Training-dependent steps"
if (Test-Path resnet\resNet34.pth) {
    Write-Host "Weights found. You can run:"
    Write-Host "  cd resnet; python evaluate.py --data-dir ..\dataset\test --weights resNet34.pth"
    Write-Host "  cd ..\grad_cam; python resnet_grad_cam.py --image-path ..\dataset\test\0\<image>.png"
} else {
    Write-Host "No trained weights found. Evaluation and Grad-CAM are ready but skipped."
}

Write-Host ""
Write-Host "Non-training workflow completed."
