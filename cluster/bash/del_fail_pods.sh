kubectl get pods | grep k5wang | grep Error | awk '{print $1}' | xargs kubectl delete pod
kubectl delete jobs k5wang-job-train
kubectl delete jobs k5wang-job-search