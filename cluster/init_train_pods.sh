pwd
kubectl create -f job_train.yaml
sleep 10
kubectl get pods | grep k5wang
sleep 10
kubectl get pods | grep k5wang-job | awk '{print $1}' | xargs kubectl logs 
