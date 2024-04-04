# usage: sh emulate.sh 40 7040 192.168.1.10 6001
# 40:latency(RTT) in ms, 7040:bw in kbps, 192.168.1.10: client IP, 6001: serverPort
# For latency, each way (downlink/uplink) is added half of the additional latency
# We use ip1/port1 as the destination IP/port to identify the first TCP flow
#ip1=`cat ./pipe1IP.txt`
#port1=`cat ./pipe1Port.txt`
ip1=$3
port1=$4
# delay1 is the total addtional latency added to RTT of the first TCP flow
delay1=$(($1/2))
# bw1 is the bandwidth of network1 for first TCP flow
bw1=$2
# enable ifb kernel module for traffic shaping of incoming traffic
sudo modprobe ifb numifbs=4
echo "Add $delay1 ms delay for $ip1:$port1 on interface eno1."
# setup four virtual interface
sudo ip link set dev ifb1 up
sudo ip link set dev ifb3 up
sudo tc qdisc del dev eno1 root
sudo tc qdisc del dev eno1 ingress
# classify outgoing traffic
sudo tc qdisc add dev eno1 handle 1: root htb
sudo tc class add dev eno1 parent 1: classid 1:1 htb rate 1000Mbps
sudo tc class add dev eno1 parent 1:1 classid 1:11 htb rate 1000Mbps
# Use netem to add delay
sudo tc qdisc add dev eno1 parent 1:11 handle 10: netem delay $delay1"ms"
# outgoing traffic of first TCP flow goes to ifb1 for bandwidth throttling
sudo tc filter add dev eno1 protocol ip prio 1 u32 match ip dst $ip1 match ip sport $port1 0xffff flowid 1:11 action mirred egress redirect dev ifb1
# actual bandwidth throttling rule
sudo tc qdisc del dev ifb1 root
sudo tc qdisc add dev ifb1 handle 1: root tbf rate $bw1"kbit" burst 20k latency 1ms
# traffic shaping of incoming traffic of two flows.  Use ifb3.
sudo tc qdisc add dev eno1 handle ffff: ingress
sudo tc filter add dev eno1 parent ffff: protocol ip u32 match ip src $ip1 match ip dport $port1 0xffff action mirred egress redirect dev ifb3
# only add delays for incoming traffic
sudo tc qdisc del dev ifb3 root
sudo tc qdisc add dev ifb3 handle 1: root netem delay $delay1"ms"
sudo tc -s qdisc ls dev eno1
sudo tc -s qdisc ls dev ifb1
sudo tc -s qdisc ls dev ifb3
echo "finish."

