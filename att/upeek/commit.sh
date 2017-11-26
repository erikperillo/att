#!/bin/bash

set -ex

#name of model
name="$(cat NAME)"
#type of version bump. can be major, minor, patch
bump_type=$1
#extension of git commit message
git_msg=$2

[[ -z $bump_type ]] && { echo "error: must specify version bump type"; exit 1; }

#bumping version
version=$(./bump_version.sh "$(cat VERSION)" $bump_type)
echo "version: $version"
echo $version > VERSION

#tag it
git add -A
git commit -m "[$name version $version] $git_msg"
git tag -a "$name-$version" -m "[$name version $version] $git_msg"
