import cv2
import numpy as np
import json
from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES


'''
Script para contagem de veículos em vídeo usando RF-DETR (https://github.com/roboflow/rf-detr).
O vídeo é processado em 360p, e os veículos são contados quando cruzam uma linha central.
As contagens são salvas em um arquivo JSON e o vídeo processado é salvo em MP4.
O script utiliza a biblioteca OpenCV para manipulação de vídeo e a biblioteca RF-DETR para detecção de objetos.
'''


# Inicializa modelo
model = RFDETRBase()

# Caminho do vídeo
video_path = "Traffic IP Camera video.mkv"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

# Parâmetros do vídeo
resize_width, resize_height = 640, 360  # 360p
frame_skip = 5  # pula 5 frames
line_y = int(resize_height * 0.6)  # linha central pra contagem
offset = 50  # distância entre a linha central e as auxiliares
line_y_top = line_y - offset
line_y_bottom = line_y + offset

# Configuração do VideoWriter para salvar o resultado
output_path = "output_video.mp4"
original_fps = cap.get(cv2.CAP_PROP_FPS)
fps = original_fps / frame_skip  # Ajusta FPS pelo frame_skip
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (resize_width, resize_height))

# Contadores
counter = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
next_object_id = 0
tracked_objects = {}

# Função pra calcular distância euclidiana entre dois pontos
def euclidean(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1
    if frame_index % frame_skip != 0:
        continue

    frame_resized = cv2.resize(frame, (resize_width, resize_height))
    image_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
    detections = model.predict(image_pil, threshold=0.5)

    # Processa detecções
    for box, class_id in zip(detections.xyxy, detections.class_id): # Obtem as coordenadas do bounding box e o id da classe (carro, moto, etc.)
        class_name = COCO_CLASSES[class_id]
        if class_name not in counter:
            continue

        x1, y1, x2, y2 = box
        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y2) / 2)
        centroid = (x_center, y_center) #calcula o centroide do objeto

        # Faz o rastreamento simples dos objetos:
        # Para cada detecção atual, verifica se há um objeto semelhante (mesma classe)
        # já rastreado em frames anteriores, comparando os centroides.
        # Se a distância for pequena entre o frame passado e o atual, o movimento estiver na direção esperada (ex: para baixo, coordenada y),
        # assume-se que é o mesmo objeto e associa com o ID do frame anterior.
        matched_id = None
        min_distance = float('inf')
        
        # Verifica se o objeto já está rastreado
        for obj_id, data in tracked_objects.items():
            distance = euclidean(data['centroid'], centroid)
            # Se a classe é amesma e a distância é pequena e o objeto se desloca pra baixo, é mapeado como o mesmo objeto
            if (data['class'] == class_name and distance < 50 and 
                (centroid[1] > data['centroid'][1] or distance < 30)): 
                if distance < min_distance:
                    min_distance = distance
                    matched_id = obj_id

        # Se não encontrou objeto correspondente, cria um novo
        if matched_id is None:
            matched_id = next_object_id
            tracked_objects[matched_id] = {
                'class': class_name,
                'centroid': centroid,
                'state': 'none',
                'frames_since_last_seen': 0,
                'prev_centroid': centroid
            }
            next_object_id += 1 # Incrementa o ID para o próximo objeto
        else:
            tracked_objects[matched_id]['prev_centroid'] = tracked_objects[matched_id]['centroid']
            tracked_objects[matched_id]['centroid'] = centroid
            tracked_objects[matched_id]['frames_since_last_seen'] = 0

        # Atualiza o estado do objeto
        prev_state = tracked_objects[matched_id]['state']
        prev_centroid = tracked_objects[matched_id]['prev_centroid']

        # Lógica de contagem
        if prev_state != 'counted': # garante que cada objeto só seja contado uma vez
            # Para cada objeto rastreado, se estava acima da linha central e agora está abaixo, pontua a contagem pra classe correspondente
            if prev_centroid[1] < line_y and y_center >= line_y:
                tracked_objects[matched_id]['state'] = 'counted'
                counter[class_name] += 1
                print(f"Contado {class_name} - Total: {counter[class_name]}") 

        # Desenha o bounding box e centroide
        color = (0, 255, 0) if tracked_objects[matched_id]['state'] != "counted" else (0, 0, 255)
        cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.circle(frame_resized, (x_center, y_center), 4, color, -1)

        label = f"{class_name}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_resized, (int(x1), int(y1) - text_height - 10), (int(x1) + text_width, int(y1)), color, -1)
        cv2.putText(frame_resized, label, (int(x1), int(y1) - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Remove objetos não vistos há mais de 5 frames processados pra evitar sobrecarga de memória
    to_delete = [obj_id for obj_id, data in tracked_objects.items() 
                if data['frames_since_last_seen'] > 5]
    for obj_id in to_delete:
        del tracked_objects[obj_id]

    # Incrementa contador da ultima vez visto para todos os objetos
    for obj_id in tracked_objects:
        tracked_objects[obj_id]['frames_since_last_seen'] += 1

    # Desenha as linhas de referência
    cv2.line(frame_resized, (0, line_y_top), (resize_width, line_y_top), (255, 0, 0), 1)
    cv2.line(frame_resized, (0, line_y), (resize_width, line_y), (0, 255, 255), 2)
    cv2.line(frame_resized, (0, line_y_bottom), (resize_width, line_y_bottom), (0, 0, 255), 1)

    # Exibe contagem na tela
    for i, (class_name, count) in enumerate(counter.items()):
        cv2.putText(frame_resized, f"{class_name}: {count}", (10, 30 + i*30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Grava o frame processado
    out.write(frame_resized)

    # Exibe o vídeo
    cv2.imshow("Detecção RF-DETR 360p", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release() 
cv2.destroyAllWindows()

# Salva JSON com contagem final
with open("contagem_total.json", "w") as f:
    json.dump(counter, f, indent=4)

print("Contagem final:")
print(counter)
print(f"Vídeo processado salvo em: {output_path}")
