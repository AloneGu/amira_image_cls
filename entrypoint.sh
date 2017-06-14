#!/bin/sh

thisdir=$(dirname $(readlink -f "$0"))
cd $thisdir

uwsgi --master --need-app --http 0.0.0.0:8090 \
    --module img_cls.apps --callable app \
    --processes 1 --threads 1 \
    --logto2 /data/log/api.log \
    --touch-logrotate /tmp/api-logrotate
