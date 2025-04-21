---
title: Scheduling
summary: The scheduling configuration flags can be configured via command line flags or environment variables.
---

# Scheduling Configuration Flags

The Scheduling node is used for providing the `scheduling` microservice for PD. You can configure it using command-line flags or environment variables.

## `--advertise-listen-addr`

- The URL for the client to access the Scheduling node.
- Default: `${listen-addr}`
- In Docker or NAT network environments, if a client cannot access the Scheduling node through the default client URLs listened to by `scheduling`, you must manually set `--advertise-listen-addr` for client access.
- For instance, the internal IP address of Docker is `172.17.0.1`, while the IP address of the host is `192.168.100.113` and the port mapping is set to `-p 3379:3379`. Given this, you can set `--advertise-listen-addr="http://192.168.100.113:3379"`. Then, the client can find this service through `http://192.168.100.113:3379`.
