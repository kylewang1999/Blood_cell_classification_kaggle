
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
    A peek inside storage.yaml:
    ```yaml
    apiVersion: v1
    kind: PersistentVolumeClaim
    metadata:
      name: <your-volume-name>
    spec:
      storageClassName: rook-cephfs
      accessModes:
      - ReadWriteMany
      resources:
        requests:
          storage: 500Gi
    ```


2. Check PVC(Persistent volume claim)

    ```bash
    kubectl get pvc ${VOLUME_NAME}
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
4. Display content printed to the console
   ```
   kubectl logs ${POD/JOB_NAME}
   ```
5. Delete Pod/Job with ```kubectl delete pod/job ${pod/job_name}```

### a) Start a bash shell in the pod/job
```bash
kubectl exec -it ${POD_NAME} bash
```
Terimnate shell with ```exit```

### b) Deploy the job
```bash
kubectl apply -f <job_name>.yaml      # or
kubectl create -f <job_name>.yaml
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

## Exmaple of login.yaml
> This is a Pod. ***NEVER*** use ```args: ["-c", "sleep infinity"]``` for a Job
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: haoban-login
spec:
  containers:
  - name: vol-container
    image: gitlab-registry.nautilus.optiputer.net/vamsirk/research-containers
    command: ["/bin/bash"]
    args: ["-c", "sleep infinity"]
    resources:
      requests:
        memory: "8Gi"
        cpu: 2
      limits:
        nvidia.com/gpu: 1
        memory: "8Gi"
        cpu: 2
    volumeMounts:
    - name: haoban-volume     #use your own volune path
      mountPath: /haoban-volume
  restartPolicy: Never
  volumes:
    - name: haoban-volume
      persistentVolumeClaim:
        claimName: haoban-volume
  nodeSelector:
    gpu-type: "1080Ti"
```

## Useful bash commands & more
[**Set alias to commands:**](https://linuxize.com/post/how-to-create-bash-aliases/)
  ```bash
  alias alias_name="command_to_run"
  ```
[**Delete failed pods**](https://gist.github.com/zparnold/0e72d7d3563da2704b900e3b953a8229): 
  ```bash
  kubectl get pods | grep k5wang |grep Error | awk '{print $1}' | xargs kubectl delete pod
  ```
[**Sleep**](https://www.cyberciti.biz/faq/linux-unix-sleep-bash-scripting/) (Wait before executing next line of bash script): 
  ```sleep ${DURATION}```

[**Run multiple bash command with .yaml file**](https://stackoverflow.com/questions/33887194/how-to-set-multiple-commands-in-one-yaml-file-with-kubernetes):
  ```yaml
  command: ["/bin/bash","-c"]
  args: ["${COMMAND_1}; ${COMMAN_2}"]
  ```
**Easier pod/job deletion with labels:**
Executing a job will likely create multiple pods with non-human-friendly suffixes. It is therefore easier to set a label for job (which is inherited by the spanwed pods), and use that as an identifier.
  - *Step1*: [Set a label for the pod/job](https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/)
    ```yaml
    metadata:
      name: name-of-your-pod
      labels:
        stage: train
        model: CIFAR10
        data: blood_cell
    ```
  - *Step2*: [Use lables to delete desired pods/jobs](https://stackoverflow.com/questions/59473707/kubenetes-pod-delete-with-pattern-match-or-wilcard)
    ```bash
    kubectl delete pods -l ${label_key}=${label_value}
    ```
