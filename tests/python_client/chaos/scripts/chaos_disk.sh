
nodes=("10.100.32.154", "10.100.32.155", '10.100.32.147')
for node in ${nodes[*]}
do
  echo "start to shh node $node"
  bash ssh -i .ssh/k8sKey zilliz@$node

  echo "start to disk attack read payload"
  bash chaosd attack disk add-payload read -s 1000G -n 7 -p /dev/zero
done