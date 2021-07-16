# Keeps track of TODO's and BUGs

## TODO

1. ~~Figure out how to deploy using kubectl~~

2. ~~Setup bash aliases~~

3. Fix training with auxiliary enabled

4. Perform NAS from scratch

5. Visualize Training with tensorboard

## Bug Journal

1. When working locally with pathlib's list(Path.glob()), the path needs to be global path. Otherwise will produce empty list.

2. In utils, replace view() with reshape(). This is a pytorch version issue

3. In utils.load(): If using CPU, need to replace model.load_state_dict w/ torch.load()

4. [CUDA out of memory](https://pytorch.org/docs/stable/notes/faq.html) -- batch size too large

5. CUDA out of memory after changing batch size, while testing on colab - Loop thru test_queue inside of torch.no_grad()

6. In train(), infer(), change top 5 to top2, since there are only 4 clases to predict.

7. error: unrecognized arguments while running train.py with cluster/train.sh bash script. Quick fix: execute using .yaml file instead. 

8. Upon restart, import numpy in .py files result in ```ModuleNotFoundError: No module named 'numpy'```. [Stackoverflow](https://stackoverflow.com/a/40186317): 

> a) Make sure you are choosing the anaconda python interpreter
> b) Try ```conda install numpy``` if step 1 doesn't solve the problem 

9. Training with auxiliary enabled result in tensor size mismatch: ```mat1 and mat2 shapes cannot be multiplied (10x62208 and 768x4)```.
> See [this discussion](https://stackoverflow.com/a/66338440)
> Also see [discussion about 1*1 convolution](https://stats.stackexchange.com/questions/194142/what-does-1x1-convolution-mean-in-a-neural-network)
```
```




 