# nautilus cluster guide
This are some sample usages of the nautilus. You may also refer to the [Official Documents](https://gitlab.com/ucsd-prp/ucsd-prp.gitlab.io/-/tree/master/_userdocs)

# Get started
In general, we have two steps to start to use our cluster
1. *(New user only)* Apply for storage to store our files and data
2. create a pods and run your programe. A Kubernetes pod is a group of containers that are deployed together on the same host. If you frequently deploy single containers, you can generally replace the word "pod" with "container" and accurately understand the concept.

There are generally three types of pods under our situation. 
1. An independent pods is limited to exit for 6 hours
2. Pods controlled by jobs. A Job creates one or more Pods and ensures that a specified number of them successfully terminate. As pods successfully complete, the Job tracks the successful completions. When a specified number of successful completions is reached, the task (ie, Job) is complete. Deleting a Job will clean up the Pods it created.
3. Pods controlled by deployment. In case you need to have an idle pod in the cluster, that might ocassionally do some computations, you have to run it as a Deployment. Deployments in Nautilus are limited to 2 weeks (unless the namespace is added to exceptions list and runs a permanent service). This ensures your pod will not run in the cluster forever when you don't need it and move on to other projects. **Such a deployment can not request a GPU.**

So you can choose one of them based on your type of task.

## How to get the permanent storage:
There are multiple types of storage described in our documentation, and there's no one-fit-all solution.
This is sample to apply for storage. It's perfectly fine to request more than 1TB of storage if needed, and clean it up after. 
```bash
kubectl apply -f storage.yaml
```
### Checking my PVC(Persistent Volume Claim)
```
kubectl get pvc k5wang-volume
```

[Official usage](https://gitlab.com/ucsd-prp/ucsd-prp.gitlab.io/-/tree/master/_userdocs%2Fstorage)
## How to run a pods:
Use the "Pod". For pods we can use **args: ["sleep", "infinity"]**, since it will be automaticly deleted after 6 hours.
Pods can maximally exist for 6 hours. For long time use, you should create a job.
Create "Pod"
```
kubectl create -n ecepxie -f login.yaml
```
You can also manually clean you pods when it is done.
```
kubectl delete pods ${POD_NAME}
```
## How to depoloy a job
Use the "Job". For Job **Please never use args: ["sleep", "infinity"]**. This command will occupy the resources forever and can not be cleaned if your are done. I suggest you to run your code with a .sh file that set up the environment(with a conda environment file [env](environment.yml)) and specify your command in a general file. A sample is [shell](train.sh)

```
kubectl apply -f job.yaml
```
You can manually clean you job when it is done(Completed). OR job will be cleaned after a default set of 7 days.
```
kubectl delete job ${JOB_NAME}
```

## How to observe the situation of your task?
1. Get pods/job name and status
```
kubectl get pods/job
```
2. See why the pods creation is pending. This command will tell you what is wrong whe pendding happens(resouece shortage, etc.)
```
kubectl describe pods ${POD_NAME}
```
3. Logs the print for the pods you create. If you have print some information (loss, epoch, etc.) in your python file, this command can show them on your screen
```
kubectl logs ${POD_NAME}
```
4. Tensorboard. We can observe the training process with the help of tensorcoard. You should first activate the tensorboard in the pods
```
tensorboard --logdir=${LOG-FILE}
```
The kubectl can link you local port to the specified port of the pods
```
kubectl port-forward ${POD_NAME} ${REMOTE-PORTNUM}:${LOCAL-PORTNUM}
```
Then the website for tensorboard can be seen in **http://localhost:${LOCAL-PORTNUM}**
## How to use shell on the cluster
Yes, you can also use shell on the cluster. Use the following command will help you log on the pods and you can use shell there 
```
kubectl exec -it ${POD_NAME} bash
```

## How to transfer file to the cluster
For me there are two ways to transfer file to the cluster
1. kubectl copy
```
kubectl cp ${LOCAL-DIR} ${NAMESPACE-NAME}/${POD_NAME}:${REMOTE-DIR}
```
1. Git (suggested for code transfer). You can git push your code on the repo and pull them on the cluster. This method can additionally maintain history of your code, and increase efficiency if collaboration is needed.

