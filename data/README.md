# Data Notes

Place raw datasets in this folder when running real-data exploration:

- `all_stream_info.csv`
- `points_transaction_log.csv`
- `KuaiLive/` (directory containing `room.csv`, `user.csv`, `streamer.csv`, `click.csv`, `comment.csv`, `like.csv`, `gift.csv`)

Suggested policy:

- keep raw files unmodified,
- add any cleaned/derived outputs under `results/tables/`,
- avoid committing sensitive or private user data.
