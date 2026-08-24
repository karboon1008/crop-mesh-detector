# Why the Docker `coordinator` Exists (and Why It Isn't Part of the Mesh Architecture)

## Question

The `docker/` stack includes a `coordinator` service. Is this a central
server that contradicts the project's "decentralised mesh" design?

## Short answer

No. The coordinator is a **deployment-layer convenience introduced by the
Docker simulation**, not a requirement of the mesh's learning architecture.
It exists purely to solve a round-synchronization problem created by running
each node as a separate container/process — it carries no learning-relevant
responsibility (no model weights, no knowledge payloads, no aggregation, no
central metrics store).

## Evidence

1. **The non-Docker path has no coordinator at all.**
   `src/federated/mesh.py`'s `MeshSimulator.run_round()` orchestrates rounds
   as a plain in-process Python loop over nodes — there is no separate
   "coordinator" entity when everything runs in one process. The coordinator
   only appears once nodes are split into independent Docker containers with
   no shared memory or clock.

2. **The coordinator only schedules; it never touches knowledge or metrics.**
   From `docker/coordinator/main.py` and `coordinator_runner.py`:
   - It fans out `POST /round/start` and `POST /round/gather` to all nodes
     concurrently (`asyncio.gather` + `httpx`), purely to keep rounds in
     lockstep.
   - It **never calls a node's `/knowledge` endpoint** — nodes fetch peer
     knowledge directly from each other. The coordinator never sees payload
     content.
   - It **owns no metrics data** — no round_metrics DB, no `merged.db`. Every
     node writes its own energy/accuracy/communication numbers into its own
     SQLite db. The coordinator's only disk write is a small `status.json`
     completion marker.
   - It exposes `/events` and `/log` only so the dashboard can show a feed of
     scheduling activity (which path was called, what status came back) —
     never response bodies.

3. **The design spec frames it the same way.**
   `docs/superpowers/specs/2026-08-16-docker-mqtt-mesh-design.md` explicitly
   describes the coordinator as a round-scheduling/liveness mechanism and
   contrasts it with a classic central model/parameter server, which it is
   not: aggregation/consensus (`trimmed_mean`/`krum`) happens locally inside
   each node process from peer payloads that node fetched itself.

4. **`docker/README.md` makes the same argument** in its "Why the coordinator
   isn't a central server" section: all the coordinator does is decide *when*
   a round starts — "a timing/liveness role, analogous to a
   barrier/synchronization signal." If it goes down mid-run, no node's data
   is lost, since it never held any.

## How to phrase this in the research writeup

The mesh's actual architecture — knowledge exchange, aggregation, consensus,
and metrics — is fully decentralised/peer-to-peer by design. The
`coordinator` container is a deployment-layer convenience needed only
because Docker splits nodes into independent processes with no shared
clock/state; it provides a round barrier and nothing else. In principle it
could be replaced by any other synchronization primitive (a shared round
file, gossip-based readiness, each node peer-polling the others, etc.)
without changing the mesh's learning semantics at all. Deploying to
independently-clocked real hardware (e.g. K210 devices) would still require
*some* synchronization signal, but it need not take the shape of a
centralized-looking HTTP service.

## Known limitation to disclose honestly

The coordinator is a **single point of failure for round scheduling**: if it
dies mid-run, no new rounds are triggered (though no data already produced by
the nodes is lost, since none of it lives on the coordinator). This should be
named explicitly as a limitation of the Docker demo's synchronization
mechanism, rather than implied away — "simulation convenience" does not mean
"architecturally inert."
