# Keeps track of TODO's and BUG's

## TODO

1. ~~Figure out how to deploy using kubectl~~

2. ~~Setup bash aliases~~

3. ~~Fix training with auxiliary enabled~~

4. Adapt to Hi I/O jobs to increase GPU utilization 

5. Monitor GPU Usage of Pods/Jobs
   
5. Perform NAS from scratch

6. Understand LPT Optimization Algo.

7. Derive image captioning Optim. Algo.

8. Visualize Training (Tensorboard?)

## Bug Journal

1. When working locally with pathlib's list(Path.glob()), the path needs to be global path. Otherwise will produce empty list.
   
2. ```AttributeError: 'str' object has no attribute 'size'```
    > Change view() to reshape()

3. In utils, replace view() with reshape(). This is a pytorch version issue

4. In utils.load(): If using CPU, need to replace model.load_state_dict w/ torch.load()

5. [CUDA out of memory](https://pytorch.org/docs/stable/notes/faq.html) -- batch size too large

6. CUDA out of memory after changing batch size, while testing on colab 
   > Loop thru test_queue inside of torch.no_grad()

7. In train(), infer(), change top 5 to top2, since there are only 4 clases to predict.

8. error: unrecognized arguments while running train.py with cluster/train.sh bash script. Quick fix: execute using .yaml file instead. 

9.  Upon restart, import numpy in .py files result in ```ModuleNotFoundError: No module named 'numpy'```. [Stackoverflow](https://stackoverflow.com/a/40186317): 
    > a) Make sure you are choosing the anaconda python interpreter
    > b) Try ```conda install numpy``` if step 1 doesn't solve the problem 

10. Training with auxiliary enabled result in tensor size mismatch: ```mat1 and mat2 shapes cannot be multiplied (10x62208 and 768x4)```.
    > See [this discussion](https://stackoverflow.com/a/66338440)
    > Also see [discussion about 1*1 convolution](https://stats.stackexchange.com/questions/194142/what-does-1x1-convolution-mean-in-a-neural-network)
    > 10*66208 = 10 * (9 * 9 * 768) ---> Modify input dimension at model.py line 102

    ~~self.classifier = nn.Linear(768*9*9, num_classes)~~

    ```python
    self.classifier = nn.Linear(768*9*9, num_classes)
    ```

11. First 1500 training steps runs fine until the loss fails to evaluate at the 1550th step.

    **Possible Cause 1 (Unlikely): Invalid image or label**
    > Remove image or image labels that contain ```"Nan"``` values

    **Possible Cause 2: Complex loss surface or computing zero log**
    > Reduce learning rate (network learning rate) **NOT** ~~arch_learning_rate~~
    Change loss function to ```loss = criterion(logits + 1e-12, target)```

12. train_search_ts.py ```infer()```  works fine during first 200 steps, then halts because of ```illegal memory access```
    >  Possible reason is the incompatibility of PyTorch and CUDA
    > Changing GPU device from ~~**1080Ti**~~ to **2080Ti**

13. Low GPU Utilization
    > This is due to [hi I/O](https://pacificresearchplatform.org/userdocs/running/io-jobs/) access to persistent volume claim
    