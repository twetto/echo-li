echo-li-python/venv/bin/python \
    echo-li-python/tests/diagnostics/midair_rudolf_flow_error.py \
    --root /home/twetto/Server250/18TB/datasets/dataset_MidAir/MidAir \
    --set VO_test \
    --cond sunny \
    --traj 0 \
    --frames 300 \
    --scale 0.5 \
    --config configs/diagnostics_midair_sparse3d.yaml \
    --plot-out /tmp/midair_rudolf_flow_model_sunny.png
