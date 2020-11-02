save=ende-baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 python3  train.py --ddp-backend=no_c10d data-bin/wmt16_en_de_bpe32k \
    --arch transformer_wmt_en_de  --reset-optimizer  --fp16 --share-all-embeddings \
	    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --me 30\
	      --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
	        --lr 0.0007 --min-lr 1e-09 --dropout 0.1 \
           --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0\
		    --max-tokens  4096  --save-dir checkpoints/$save \
		    --update-freq 2 --no-progress-bar --log-format json --log-interval 25 --save-interval-updates  1000 --keep-interval-updates 100 | tee out.$save
