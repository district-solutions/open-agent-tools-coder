#!/bin/bash
# shellcheck disable=SC2002,SC1091

function load_env() {
    if [[ -e ./.env ]]; then
        source ./.env
    fi
}

function deploy_app() {
    compose="./vllm/tools/functiongemma/docker-compose-functiongemma.yaml"
    container_name=$(grep container_name "${compose}" |  grep -v '# ' | awk '{printf "%s ", $NF}')
    vc="docker stop -t 0 ${container_name}"
    eval "${vc}"
    vc="docker rm -f ${container_name}"
    eval "${vc}"

    vc="docker-compose -f ${compose} up -d"
    echo "starting with:"
    echo "${vc}"
    eval "${vc}"
    lt="$?"
    if [[ "${lt}" -ne 0 ]]; then
        echo -e "failed to start with command:\n${vc}"
        exit 1
    fi
}

function main() {
    load_env
    deploy_app
}

main

exit 0
