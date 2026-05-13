#!/bin/bash
# shellcheck disable=SC2317,SC2174,SC2120,SC2119,SC2009,SC2002,SC1091

if [[ -e ./.env ]]; then
    source ./.env
fi

function deploy_primary() {
    compose_file="./vllm/chat/qwen36-35b-awq4.yaml"
    container_name=$(cat "${compose_file}" | grep container_name | awk '{print $NF}')
    echo "force shutdown"
    docker stop -t 0 "${container_name}"
    docker rm -f "${container_name}"

    vc="docker-compose -f ${compose_file} up -d"
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
    deploy_primary
}

main

exit 0
