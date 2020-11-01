#path for saving the checkpoint
save=ende-tlat
#abosulute path to the target dictionary
tgt_dict=`pwd`/data-bin/wmt16_en_de_bpe32k_pretrain/dict.de.txt

CUDA_VISIBLE_DEVICES=0,1,2,3  python3 train.py --ddp-backend=no_c10d  data-bin/wmt16_en_de_bpe32k_pretrain \
	  --arch transformer_wmt_en_de  --share-all-embeddings --fp16 \
	    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
	      --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 1000 \
	        --lr 0.000175 --min-lr 1e-09 --dropout 0.1 \
         --me 45 --reset-optimizer --reset-lr-scheduler \
         --adaptive-training --dict-file $tgt_dict --adaptive-method 'exp' --adaptive-T 3.5  --weight-drop 0.3   --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0\
		    --max-tokens  4096  --save-dir checkpoints/$save\
		    --update-freq 2 --no-progress-bar --log-format json --log-interval 25 --save-interval-updates  1000 --keep-interval-updates 100 2>&1 | tee out.$save
