
# A Guide to Delopy Through Kubectl

- Namespace: ```ecepxie```
- 1st Pod upon login: ```k5wang-login```
- Storage (volume) for codes: ```k5wang-volume```
  - Path to this project: ```/k5wang-volume/Blood_cell_classification/```
- Storage (volume) for datasets: ```k5wang-volume-datasets```
  - Path to this dataset: ```/k5wang-volume-datasets/kaggle/blood-cell```
- To remove dir [quickly](https://yonglhuang.com/rm-file/): ```find ./ -type f -delete```

## I. First Time Use  

1. Apply for permanant storage

   ```bash
   kubectl apply -f storage.yaml
   ```

2. Check PVC(Persistent volume claim)

    ```bash
    kubectl get pvc k5wang-volume
    ```

## II. Setting up pod/job for deployment

1. Create a pod/job (specified by *.yaml) 
   ```bash
   kubectl create -n ecepxie -f ../cluster/<yaml_file_name>.yaml
   ```

2. Check created pod/job status
   ```bash
    kubectl get pods/jobs ${pod/job_name_in_yaml}
    kubectl describe pods/jobs ${pod/job_name_in_yaml}
   ```
3. Check gpu resource
    ```bash
   kubectl get nodes -L gpu-type
    ```
4. Delete Pod/Job with ```kubectl delete pod/job ${pod/job_name}```

### a) Start a bash shell in the pod/job
```bash
kubectl exec -it ${POD_NAME} bash
```
and terminate it
```bash
exit
```

### b) Deploy the job
```bash
kubectl apply -f <job_name>.yaml
```

Note:
- ```kind``` in job.yaml should be ```Job```, not ```Pod``` anymore
- Never use ```args: ["sleep", "infinity"]``` for a Job

## Transfer file to the cluster
For me there are two ways to transfer file to the cluster
1. kubectl copy
```
kubectl cp ${LOCAL-DIR} ${NAMESPACE-NAME}/${POD_NAME}:${REMOTE-DIR}
```
For this blood cell classification project:
```
kubectl cp kaggle ecepxie/k5wang-login:k5wang-volume/Blood_cell_classification_kaggle
```

2. Git (suggested for code transfer). You can git push your code on the repo and pull them on the cluster. This method can additionally maintain history of your code, and increase efficiency if collaboration is needed.


- Apply a separate storage volume for datasets
