#!/usr/bin/env bash

ps -ef | grep python |grep "serving/main" | awk '{print $2}' |xargs -I{} kill {}

ps -ef | grep python |grep "serving/main" | awk '{print $2}' |xargs -I{} kill -9 {}
