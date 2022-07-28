todo:
- write tests
- make it possible for state(env) to return all states (for transformer)
- improve StrongholdReplay (probably related to previous)
  - Not fixed length (can change # steps without restarting program
  - probably all mutable
  - use an mmatrix instead of a sizedvector{svector}