
1. Build Docker Image
```bash
docker build . -t scale_rope
```

2. Launch Docker Container
```bash
docker run --gpus all -m 128g -itd -v .:/app --name scale_rope scale_rope:latest /bin/bash
```

3. Connect to Docker Container
```bash
docker exec -it scale_rope /bin/bash
```

4. In docker:
```bash
./docker_req.sh
```
