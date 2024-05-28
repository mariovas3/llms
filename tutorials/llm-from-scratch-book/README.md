# Implementing GPT-2;

*This part of the repo is motivated by Sebastian Raschka's book "Build a Large Language Model from Scratch".*


## Data:
The data are the raw text of <a href="https://en.wikisource.org/wiki/The_Verdict">"The Verdict"</a>. I used this dataset to test that my model can overfit it.

## Training:
* Open a terminal session and run `export PYTHONPATH=.` and navigate to `llms-from-scratch-book`. From there run:

	```bash
	python training/run_experiment.py --context_length=256 --max_epochs=10 --val_check_interval=5 --val_loop_iters=1 --pre_norm
	```

## Testing:
* Build the docker cpu image using:

	```bash
	DOCKER_BUILDKIT=1 docker build -t llm-from-scratch -f Dockerfile-cpu .
	```

* Run the tests:

	```bash
	docker run --rm llm-from-scratch
	```

## Lessons learned:

### MHA
* Lots of memory caching going on in parallel attention implementation as
	compared to for loop over heads implementation. Makes sense since
	each head computation is a lot less than doing everything at once.
	PyTorch and TensorFlow seem to be caching intermediate results.

### Pre-norm vs Post-norm <a href="https://arxiv.org/pdf/2002.04745">paper here</a>
* Pre-norm looks to work a lot better than post-norm.
	* 10 epochs of training with `pre-norm` led to Average batch train loss in epoch of 0.7650676237212287.
		* Given the prompt "Every effort moves you", the generation via sampling with temperature=1 is:
		> Every effort moves you?"\n\n"Yes--quite insensible to the irony. She wanted him vindicated--and by me!"\n\nHe laughed again, and threw back his head to look up at the sketch of the donkey. "There were days when I'
	* 10 epochs of training with `post-norm` led to Average batch train loss in epoch of 6.079882356855604
		* Given the prompt "Every effort moves you", the generation via sampling with temperature=1 is:
		> Every effort moves you,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

### Decoder Layer clones vs fresh inits:
* Fresh inits, as expected, worked better. Idk why the PyTorch implementation
	of `TransformerDecoder` opts in for clones of `TransformerDecoderLayer`
	as per <a href="https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py#L452">this</a>.
