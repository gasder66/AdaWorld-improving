torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    main.py fit \
    --config config/lam_a2d.yaml \
    2>&1 | tee output_train_a2d.log
