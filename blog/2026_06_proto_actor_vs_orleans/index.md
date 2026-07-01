---
title: ProtoActor vs Orleans
---

## Reentrancy

ProtoActor supports Reentrancy by a `Continuation` system message. Since system messages have higher built-in precedence continuations are executed front-of-line instead or being enqueued in user message queue.

In Orleans such a mechanism does not exist. So if a non-awaited continuation needs reentrancy it can only send a message the common way, resulting in normal enqueuing back-of-line.
