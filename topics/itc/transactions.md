---
title: Transactions
draft: true
---

Structured data at the first glance is simple. To grasp the semantics of a model, looking at its data structure is often easier than going through the logic dealing with it.

Why is this the case?

Well, we look at data either through some kind of declarative model or by example data. With good naming in place one can infer a lot of intent and consistency invariants, either expressed explicitly or intuitively.

The logic dealing with the data, supported by data model mechanism, ensure these consistency invariants.

Still sounds simple, but the difficulties show up soon when fault-tolerance and concurrency come into play.

The root of difficulty arises from sharing data in a modifiable way between multiple concurrently operating actors.

First let's assume we have some granularity of data shared which comprises invariants necessary to keep up. Let's call such a scope of data an *aggregate*, using the terminology of DDD[^2].

The simplest way to avoid consistency difficulties is the serialize units of consistency preserving modifications. Not sharing at all or not modifying at all can be seen as extreme variants of serializing. The first ensuring serialized access by separation in location and the latter by time. For this one can think of immutable data as created once at before reading it multiple times afterwards.

Striving for simplest possible solutions, going for serializing should be preferred. If possible by not sharing or not mutating at all. That's why shared-nothing and immutable architectures are so beneficial.

However, this is not always possible by the demands of the problem domain.

Anomalies[^1]:

- Dirty read
- Dirty write
- Read skew
- Phantom reads
- Lost updates
- Write skew


[^1]: Kleppman, Martin; Riccomini, Chris (2026). "Designing Data-Intensive Applications (2nd Edition)".
[^2]: Domain Driven Design
