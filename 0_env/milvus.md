# 在 Docker 中运行 Milvus (Linux)

本页说明如何在 Docker 中启动 Milvus 实例。

## 前提条件

- [安装 Docker](https://docs.docker.com/get-docker/)。
- 安装前[请检查硬件和软件要求](https://milvus.io/docs/zh/prerequisite-docker.md)。

## 在 Docker 中安装 Milvus

Milvus 提供了一个安装脚本，可将其安装为 docker 容器。该脚本可在[Milvus 存储库中](https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh)找到。要在 Docker 中安装 Milvus，只需运行

```shell
# Download the installation script
$ curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

# Start the Docker container
$ bash standalone_embed.sh start
```

**版本 2.6.6 的新功能：**

- **流节点**增强数据处理能力
- **啄木鸟 MQ**：改进了消息队列，减少了维护开销，详情请参见[使用啄木鸟](https://milvus.io/docs/zh/use-woodpecker.md)
- **优化架构**：整合组件，提高性能

请始终下载最新脚本，以确保获得最新配置和架构改进。

如果要在独立部署模式下使用[备份](https://milvus.io/docs/milvus_backup_overview.md)，建议使用[Docker Compose](https://milvus.io/docs/install_standalone-docker-compose.md)部署方法。

如果在拉取镜像时遇到任何问题，请通过[community@zilliz.com](mailto:community@zilliz.com)联系我们并提供有关问题的详细信息，我们将为您提供必要的支持。

运行安装脚本后

- 一个名为 Milvus 的 docker 容器已在**19530** 端口启动。
- 嵌入式 etcd 与 Milvus 安装在同一个容器中，服务端口为**2379**。它的配置文件被映射到当前文件夹中的**embedEtcd.yaml。**
- 要更改 Milvus 的默认配置，请将您的设置添加到当前文件夹中的**user.yaml**文件，然后重新启动服务。
- Milvus 数据卷被映射到当前文件夹中的**volumes/milvus**。

你可以访问 Milvus WebUI，网址是`http://127.0.0.1:9091/webui/` ，了解有关 Milvus 实例的更多信息。有关详细信息，请参阅[Milvus WebUI](https://milvus.io/docs/zh/milvus-webui.md)。

## （可选）更新 Milvus 配置

您可以修改当前文件夹下**user.yaml**文件中的 Milvus 配置。例如，要将`proxy.healthCheckTimeout` 更改为`1000` ms，可按如下方式修改文件：

```shell
cat << EOF > user.yaml
# Extra config to override default milvus.yaml
proxy:
  healthCheckTimeout: 1000 # ms, the interval that to do component healthy check
EOF
```

然后按如下步骤重启服务：

```shell
$ bash standalone_embed.sh restart
```

有关适用的配置项，请参阅[系统配置](https://milvus.io/docs/zh/system_configuration.md)。

## 升级 Milvus

您可以使用内置的升级命令升级到最新版本的 Milvus。它会自动下载最新配置和 Milvus 映像：

```shell
# Upgrade Milvus to the latest version
$ bash standalone_embed.sh upgrade
```

升级命令会自动

- 下载带有更新配置的最新安装脚本
- 调用最新的 Milvus Docker 映像
- 使用新版本重启容器
- 保留现有数据和配置

这是升级 Milvus 独立部署的推荐方法。

## 停止和删除 Milvus

你可以按如下方式停止和删除该容器

```shell
# Stop Milvus
$ bash standalone_embed.sh stop

# Delete Milvus data
$ bash standalone_embed.sh delete
```