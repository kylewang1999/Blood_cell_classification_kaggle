#!/bin/bash
alias kube="kubectl"
alias getpods="kubectl get pods | grep k5wang"
alias getjobs="kubectl get jobs | grep k5wang"
alias get="getpods && getjobs"
alias logs="kubectl logs -l name=k5wang-train"
alias kubelogin="kubectl create -f ./config/login.yaml"
alias kubelogout="kubectl delete -f ./config/login.yaml"
alias kubeexec="kubectl exec -it k5wang-login bash"
alias exe="kubeexec"
alias p="pwd"
alias chdir="cd k5wang-volume/Blood_cell_classification_kaggle/darts-LPT/"

#/k5wang-volume/Blood_cell_classification_kaggle/cluster
#/k5wang-volume/Blood_cell_classification_kaggle/darts-LPT/search-GPU1080-20210725-084552#
