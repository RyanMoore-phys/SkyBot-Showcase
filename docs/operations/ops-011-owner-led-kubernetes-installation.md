# Owner-led Kubernetes installation and verification companion

> **Document control:** OPS-011 · Operations / installation plan · **State:** Proposed; commands for future owner execution, not executed · **Custodian:** Repository maintainer · Revision history: Git.

**Date:** 2026-10-28. Owner preference: personally perform Kubernetes setup for practice. Plan: [GOV-018](../governance/gov-018-cluster-and-aiops-implementation-plan.md). Topology: [ARC-007](../architecture/arc-007-cluster-platform.md). Access handling: [OPS-008](ops-008-host-access.md).

## 1. How to use this companion

Work one stage at a time, inspect its result, and continue only when the stated check passes. All commands below are documentation examples for the owner. This task has not connected to any host or executed installation commands. Never paste credentials, join tokens, kubeconfigs, raw node addresses or private logs into public CI output.

The machines already have Linux and Kubernetes commands installed. First determine whether a cluster exists. Do not run `kubeadm init`, reinstall a runtime or reset a node merely because a tool is installed. The proposed new-cluster path is kubeadm; an existing healthy cluster may be adopted after compatibility/security checks instead.

Version placeholders are deliberate: the OS, installed versions and network ranges have not been inspected. Record them privately, select a supported mutually compatible version set, then generate/review the exact version-specific configuration before initialization. Files named below under a private operator directory or future manifests are implementation deliverables, not files provided by this documentation-only change.

## 2. Stage A — read-only inventory, on each node

Run individually on `skybot`, `node1`, `node2`, and the GPU host through the existing authorized access method:

```bash
hostname
cat /etc/os-release
uname -m
nproc
free -h
df -hT
lsblk -f
swapon --show
command -v kubeadm kubectl kubelet containerd crictl
kubeadm version -o short
kubectl version --client
kubelet --version
containerd --version
systemctl is-active kubelet containerd
```

Missing commands are inventory findings, not reasons to improvise an installer. Privately inspect network interfaces/routes, listening services, current containers and systemd units. Record existing runner, monitoring, mirror/storage and Minecraft services with their resource use and data locations. Check disk health and backups. On the GPU host also inspect `nvidia-smi` and record GPU model/count, driver and active processes. Do not stop an active trainer.

If an authorized kubeconfig already exists, inspect its context before contacting a cluster:

```bash
kubectl config current-context
kubectl config get-contexts
kubectl cluster-info
kubectl get nodes -o wide
kubectl get pods -A
kubectl get storageclass
```

**Check A:** classify each machine as existing member, fresh candidate or conflicting installation. Confirm intended cluster identity, compatible architectures, available resources and current services. If an existing cluster is usable, skip bootstrap and validate its components against the remaining stages. Never run reset/uninstall to resolve uncertainty.

## 3. Stage B — select and prepare the platform

Privately record: Kubernetes minor/patch, kubeadm/kubelet/kubectl versions, containerd version and CRI socket, CNI version/mode, GPU toolkit/plugin versions, private API endpoint, node addresses, non-overlapping pod/service CIDRs, DNS and time synchronization. LAN, tailnet, VPN, pod and service ranges must not overlap. Validate MTU for the chosen network path.

Use the current [kubeadm prerequisites and installation guide](https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/install-kubeadm/) for the actual distribution. Match kubelet/runtime cgroup drivers, enable the required CRI plugin and establish the swap policy. Proposed baseline disables swap persistently on dedicated nodes; review existing services before applying that choice. On the GPU host in particular, the trainer's resident memory reaches 11.2 GB on a 16 GB machine (ARC-007 section 4), so measure free memory and whether swap is in use before disabling swap or setting kubelet reservations there. Do not overwrite an existing containerd configuration wholesale.

For Debian/Ubuntu only, after configuring the official repository for the selected Kubernetes minor version, an exact-version installation takes this form:

```bash
sudo apt-get update
sudo apt-get install \
  kubelet="${SKYBOT_K8S_PACKAGE_VERSION:?set exact package version}" \
  kubeadm="${SKYBOT_K8S_PACKAGE_VERSION}" \
  kubectl="${SKYBOT_K8S_PACKAGE_VERSION}"
sudo apt-mark hold kubelet kubeadm kubectl
```

Do not execute this on an existing cluster as an unplanned upgrade. For other distributions use the corresponding official package procedure. Prepare forwarding/kernel settings and private firewall rules for the chosen CNI and Kubernetes ports; do not disable the firewall globally. Confirm node-to-node connectivity and a stable private control-plane endpoint. Reserve OS/control-plane resources and cap the outside-cluster runner before scheduling applications.

**Check B:** supported pinned versions, runtime healthy, correct cgroup/CRI setup, time synchronized, required private connectivity and adequate capacity. Record a maintenance window before restarting any runtime on the GPU host.

## 4. Stage C — initialize a new control plane, owner only

Proposed host is `skybot`, subject to its runner/control-plane capacity check. Prepare a private kubeadm configuration file using the selected version's supported schema. It must specify the node/API addresses, endpoint, Kubernetes version, pod/service CIDRs, CRI socket and kubelet reservations. Prepare encryption-at-rest and audit configuration with protected keys/retention, and the corresponding API-server mounts/arguments. These details require version-specific review before execution.

Use the official [cluster creation procedure](https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/create-cluster-kubeadm/) to check the configuration. The execution pattern is:

```bash
sudo kubeadm config images pull --config "${SKYBOT_KUBEADM_CONFIG:?review private config first}"
sudo kubeadm init --config "${SKYBOT_KUBEADM_CONFIG}"
```

Keep the generated admin configuration and bootstrap output private. For the owner, use a dedicated kubeconfig file rather than overwriting an existing configuration:

```bash
mkdir -p "$HOME/.kube"
chmod 700 "$HOME/.kube"
sudo install -m 600 -o "$(id -u)" -g "$(id -g)" \
  /etc/kubernetes/admin.conf "$HOME/.kube/skybot-cluster-admin.conf"
export KUBECONFIG="$HOME/.kube/skybot-cluster-admin.conf"
kubectl get nodes
kubectl get pods -n kube-system
```

The admin kubeconfig is for the owner, never CI or AIOps. A new node may remain NotReady and DNS pods Pending before CNI installation. Do not remove the control-plane taint to conceal an unrelated scheduling problem.

## 5. Stage D — network and CPU workers

Select a Calico release compatible with the chosen Kubernetes version. Download its official installation artifacts to the private operator directory, verify origin/version, inspect privileges, and configure its IP pool/mode to match the chosen CIDR and network. Use the release's reviewed install sequence; the future local artifact command pattern is:

```bash
kubectl apply -f "${SKYBOT_CNI_OPERATOR_MANIFEST:?review pinned artifact first}"
kubectl apply -f "${SKYBOT_CNI_CUSTOM_RESOURCES:?review matching network configuration}"
```

These paths must contain the complete required artifacts for that release; a newer release may require additional CRD installation steps. Follow its documented order rather than guessing. Do not copy a tutorial's default pod range onto this LAN. Reference: [Calico installation documentation](https://docs.tigera.io/calico/latest/getting-started/kubernetes/quickstart) (the quickstart; the single-node page does not describe a multi-node cluster).

On the control plane, generate a short-lived join command only when ready to join a worker:

```bash
sudo kubeadm token create --ttl 30m --print-join-command
```

Execute the resulting command privately with `sudo` on node1 and node2, selecting the reviewed CRI socket if needed. Retain CA verification; never use discovery verification bypass flags. Do not store the token in the repository. Expire/revoke unused bootstrap tokens after joining.

```bash
kubectl get nodes -o wide
kubectl get pods -A
kubectl get events -A --sort-by=.metadata.creationTimestamp
```

**Check D:** CPU nodes become Ready, cross-node DNS/service traffic works, forbidden traffic is denied by a disposable NetworkPolicy test, and control-plane taint remains. Use reviewed probe manifests supplied in CP-1; do not claim success solely from pod status. Keep existing stateful services in place until their migration work package.

## 6. Stage E — GPU worker, at a safe maintenance boundary

First stop new legacy admissions and allow the active trainer to finish. Confirm the GPU is free and checkpoints/backups are safe. Join the GPU worker with the same private short-lived-token process. Verify the existing driver before changing it. Install the pinned NVIDIA toolkit according to its [official installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

For the selected containerd integration, the documented configuration pattern is below; review how it affects the existing runtime configuration first:

```bash
sudo nvidia-ctk runtime configure --runtime=containerd
sudo systemctl restart containerd
```

Specify/verify an NVIDIA RuntimeClass or the validated runtime default according to the selected toolkit/plugin integration. Deploy the pinned [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin) using reviewed local manifests, with GPU-node selection and the necessary tolerations. No time slicing or shared-GPU configuration in this release.

Apply role labels after confirming the actual Kubernetes node names. Examples assume the inventory confirmed these names:

```bash
kubectl label node main skybot.dev/role=gpu --overwrite
kubectl label node skybot node1 node2 skybot.dev/role=cpu --overwrite
kubectl taint node main skybot.dev/gpu-only=true:NoSchedule --overwrite
kubectl get node main -o jsonpath='{.status.allocatable.nvidia\.com/gpu}'
```

The GPU taint excludes general workloads; trainer and required GPU-system manifests need a matching toleration. Apply it before general workload admission. Review existing pods first: NoSchedule does not evict pods already running. Calico and other node agents also need appropriate scheduling rules.

Run a reviewed disposable CUDA smoke Job requesting exactly one GPU, then submit a second while the first is held. Verify the second cannot execute until the first releases the allocation. Preserve logs and clean up only the test resources. **Check E:** one exclusive training slot works and no legacy GPU process competes with it.

## 7. Stage F — secure application foundation

The following paths represent reviewed manifests delivered in later work packages. Apply them with the owner identity, in the supplied dependency order; a missing file means the work package is not ready, not that a manifest should be improvised on the live cluster.

```bash
kubectl diff -f "${SKYBOT_PLATFORM_MANIFEST_DIR:?review implementation artifacts first}"
kubectl apply -f "${SKYBOT_PLATFORM_MANIFEST_DIR}"
kubectl get namespaces
kubectl get resourcequota,limitrange -A
kubectl get networkpolicy -A
kubectl get storageclass,pv
kubectl get pvc -A
```

The bundle must define application namespaces, restricted pod policies, narrow service accounts/RBAC, quotas, network allowlists, retained storage and the authenticated API/gateway identities. Prove denial using test pods and identity impersonation where appropriate. For example, with the final observer service-account name:

```bash
kubectl auth can-i get secrets -n skybot-aiops \
  --as=system:serviceaccount:skybot-aiops:observer
kubectl auth can-i create pods/exec -n skybot-training \
  --as=system:serviceaccount:skybot-aiops:observer
```

Both must return `no`. These RBAC checks supplement actual network and admission tests. Install the pinned Kueue release and reviewed queue policy only after its dependencies/security settings are checked. Verify preemption is disabled and repeat the exclusive/priority tests through the real coordinator before enabling production admission.

## 8. Stage G — applications, acceptance and ownership handoff

Follow CP-2 through CP-9 for trainer, storage, Minecraft, telemetry, gateway and observer installation. Enable no paid API traffic before the budget tests pass. Useful inspection commands after deployment are:

```bash
kubectl get jobs,pods -n skybot-training
kubectl get workloads -A
kubectl get pods -n skybot-aiops
kubectl get events -n skybot-training --sort-by=.metadata.creationTimestamp
```

Inspect logs only for the selected pod/container and bounded time window; sanitize before sharing. `kubectl top` requires a deployed metrics API and is not assumed to work on a fresh cluster. For a Pending pod, inspect scheduling events, requests, taints, affinity, PVC binding and queue admission separately.

**Check G:** [ASS-021](../assurance/ass-021-cluster-and-aiops-acceptance-plan.md) gates pass, backups restore and the owner can explain each node role and inspect a queued run. Only then activate automatic CI submission under [OPS-009](ops-009-cluster-delivery-and-operations.md).

## 9. Stop conditions and recovery

Stop the current stage if it targets an unexpected cluster, conflicts with an existing node installation, would interrupt live training, lacks a verified backup, has uncertain CIDRs/versions or cannot enforce security policy. Record the exact sanitized failure and resolve it at that stage. Do not run `kubeadm reset`, delete PVCs, flush firewall rules or reinstall drivers as a generic repair.

For recovery, use [OPS-010](ops-010-cluster-storage-and-recovery.md). At each stage retain the private configuration and before/after inventory. The owner should receive concrete version-specific instructions for the next stage once its prerequisites are known; this proposal does not pretend that unknown installation values have already been resolved.

<!-- backlinks:start (generated by tools/doc_backlinks.py; do not edit) -->
## Referenced by

- [GOV-017](../governance/gov-017-cluster-and-aiops-proposal.md): Containerization, Kubernetes and AIOps proposal
<!-- backlinks:end -->
