@echo off
call conda activate yolo_tutorial
set LOCAL_FILES_SERVING_ENABLED=true
set LOCAL_FILES_DOCUMENT_ROOT=datasets
label-studio
