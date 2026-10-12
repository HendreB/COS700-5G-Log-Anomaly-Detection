# Log Data

The log files are not stored in git (they are large). Place them here:

| File | Network function | Description |
|------|------------------|-------------|
| `amf_normal_logs.txt` | AMF | Access and Mobility Management, normal traffic |
| `smf_normal_logs.txt` | SMF | Session Management, normal traffic |
| `upf_normal_logs.txt` | UPF | User Plane Function, normal traffic |
| `amf_attack_logs.txt` | AMF | Captured during the attack |
| `smf_attack_logs.txt` | SMF | Captured during the attack |
| `upf_attack_logs.txt` | UPF | Captured during the attack |
| `attack_timeline.log` | – | `ATTACK_*_START` / `ATTACK_*_STOP` times of the attack |

All logs are OpenAirInterface 5G core logs (`docker logs oai-amf / oai-smf / oai-upf`) captured in an isolated
lab, with UERANSIM simulating the gNodeB and UEs. The normal and attack captures come from the same setup on the
same day; the attack is an invalid-IMSI registration storm (30 unregistered UEs, one every 2 seconds).

Only `.txt` files directly in this folder are loaded, so older captures can be kept in subfolders.
