#!/bin/sh

set -e

error()
{
    echo "error: "$1
    exit 1
}

version=$1
type=$2

[ -z "$version" ] && { error "must provide version and bump type"; }
[ -z "$type" ] && { error "must provide version and bump type"; }

major=$(echo "$version" | cut -f1 -d.)
minor=$(echo "$version" | cut -f2 -d.)
patch=$(echo "$version" | cut -f3 -d.)

if [ "$type" = "patch" ]; then
    patch=$(($patch + 1))
elif [ "$type" = "minor" ]; then
    minor=$(($minor + 1))
elif [ "$type" = "major" ]; then
    major=$(($major + 1))
else
    error "version must be one in {patch, minor, major}"
fi

echo $major.$minor.$patch
