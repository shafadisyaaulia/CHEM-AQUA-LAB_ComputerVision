"""
CHEM-AQUA LAB: Gesture-Controlled Virtual Chemistry Lab
Proyek UAS Visi Komputer - Simulasi Pencampuran Bahan Kimia Real-time

Fitur:
- MediaPipe Hands untuk deteksi gesture
- Sistem pencampuran bahan kimia dengan perhitungan pH real-time
- 8+ bahan kimia berbeda (asam, basa, garam, indikator)
- Visualisasi hasil campuran dengan nama senyawa
- Particle effects dan edge detection
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math
import time

# ============================================================================
# GLOBAL VARIABLES - SIMULATION STATE
# ============================================================================
rotation_angle = 0
reaction_particles = []
cursor_history = deque(maxlen=7)

# Tracking variables
last_click_time = 0
click_cooldown = 0.5

# ============================================================================
# DATABASE BAHAN KIMIA
# ============================================================================
CHEMICALS_DATABASE = {
    "HCl": {
        "name": "Asam Klorida (HCl)",
        "type": "acid",
        "pH": 1.0,
        "color": (0, 0, 200),  # Dark Red
        "description": "Asam kuat"
    },
    "H2SO4": {
        "name": "Asam Sulfat (H₂SO₄)",
        "type": "acid",
        "pH": 1.5,
        "color": (0, 0, 255),  # Red
        "description": "Asam kuat korosif"
    },
    "CH3COOH": {
        "name": "Asam Asetat (CH₃COOH)",
        "type": "weak_acid",
        "pH": 4.5,
        "color": (0, 100, 255),  # Orange
        "description": "Asam lemah (cuka)"
    },
    "H2O": {
        "name": "Air (H₂O)",
        "type": "neutral",
        "pH": 7.0,
        "color": (200, 200, 200),  # Gray
        "description": "Netral"
    },
    "NaOH": {
        "name": "Natrium Hidroksida (NaOH)",
        "type": "base",
        "pH": 13.0,
        "color": (255, 0, 0),  # Blue
        "description": "Basa kuat"
    },
    "NH3": {
        "name": "Amonia (NH₃)",
        "type": "weak_base",
        "pH": 11.0,
        "color": (255, 100, 0),  # Light Blue
        "description": "Basa lemah"
    },
    "NaCl": {
        "name": "Garam Dapur (NaCl)",
        "type": "salt",
        "pH": 7.0,
        "color": (255, 255, 255),  # White
        "description": "Garam netral"
    },
    "PP": {
        "name": "Fenolftalein (PP)",
        "type": "indicator",
        "pH": 7.0,
        "color": (200, 0, 200),  # Pink
        "description": "Indikator basa"
    }
}

# ============================================================================
# MIXTURE STATE - Sistem Pencampuran
# ============================================================================
class MixtureState:
    def __init__(self):
        self.components = []  # List of (chemical_id, volume)
        self.total_volume = 0
        self.current_pH = 7.0
        self.current_color = (200, 200, 200)
        self.reaction_name = "Wadah Kosong"
        self.description = "Belum ada bahan"
        
    def add_chemical(self, chemical_id, volume=10):
        """Tambah bahan kimia ke campuran"""
        if chemical_id in CHEMICALS_DATABASE:
            self.components.append((chemical_id, volume))
            self.total_volume += volume
            self._calculate_mixture()
            return True
        return False
    
    def _calculate_mixture(self):
        """Hitung pH dan warna hasil campuran"""
        if not self.components:
            self.current_pH = 7.0
            self.current_color = (200, 200, 200)
            self.reaction_name = "Wadah Kosong"
            self.description = "Belum ada bahan"
            return
        
        # Hitung weighted average pH
        total_h_concentration = 0
        total_oh_concentration = 0
        
        for chem_id, volume in self.components:
            chem = CHEMICALS_DATABASE[chem_id]
            pH = chem["pH"]
            
            # Konversi pH ke konsentrasi H+ atau OH-
            if pH < 7:
                h_conc = 10 ** (-pH) * volume
                total_h_concentration += h_conc
            elif pH > 7:
                oh_conc = 10 ** (-(14 - pH)) * volume
                total_oh_concentration += oh_conc
        
        # Hitung pH final
        if total_h_concentration > total_oh_concentration:
            net_h = total_h_concentration - total_oh_concentration
            self.current_pH = -math.log10(net_h / self.total_volume)
        elif total_oh_concentration > total_h_concentration:
            net_oh = total_oh_concentration - total_h_concentration
            pOH = -math.log10(net_oh / self.total_volume)
            self.current_pH = 14 - pOH
        else:
            self.current_pH = 7.0
        
        # Clamp pH ke range valid
        self.current_pH = max(0, min(14, self.current_pH))
        
        # Hitung warna campuran (weighted average)
        r_total, g_total, b_total = 0, 0, 0
        for chem_id, volume in self.components:
            color = CHEMICALS_DATABASE[chem_id]["color"]
            weight = volume / self.total_volume
            r_total += color[2] * weight
            g_total += color[1] * weight
            b_total += color[0] * weight
        
        self.current_color = (int(b_total), int(g_total), int(r_total))
        
        # Generate nama campuran
        self._generate_mixture_name()
    
    def _generate_mixture_name(self):
        """Generate nama dan deskripsi hasil campuran"""
        if len(self.components) == 0:
            self.reaction_name = "Wadah Kosong"
            self.description = "Belum ada bahan"
            return
        
        # Deteksi reaksi spesifik
        chem_ids = [c[0] for c in self.components]
        
        # Netralisasi asam-basa
        has_acid = any(CHEMICALS_DATABASE[c]["type"] in ["acid", "weak_acid"] for c in chem_ids)
        has_base = any(CHEMICALS_DATABASE[c]["type"] in ["base", "weak_base"] for c in chem_ids)
        
        if has_acid and has_base:
            if abs(self.current_pH - 7.0) < 0.5:
                self.reaction_name = "Reaksi Netralisasi Berhasil"
                self.description = f"Garam + Air (pH {self.current_pH:.1f})"
            elif self.current_pH < 7:
                self.reaction_name = "Campuran Asam Berlebih"
                self.description = f"Masih bersifat asam (pH {self.current_pH:.1f})"
            else:
                self.reaction_name = "Campuran Basa Berlebih"
                self.description = f"Masih bersifat basa (pH {self.current_pH:.1f})"
        
        # Campuran dengan indikator
        elif "PP" in chem_ids:
            if self.current_pH > 8.3:
                self.reaction_name = "Fenolftalein + Basa"
                self.description = "Warna merah muda (basa terdeteksi)"
            else:
                self.reaction_name = "Fenolftalein + Asam/Netral"
                self.description = "Tidak berwarna"
        
        # Campuran tunggal
        elif len(self.components) == 1:
            chem_id = self.components[0][0]
            self.reaction_name = CHEMICALS_DATABASE[chem_id]["name"]
            self.description = CHEMICALS_DATABASE[chem_id]["description"]
        
        # Campuran umum
        else:
            names = [CHEMICALS_DATABASE[c[0]]["name"].split("(")[0].strip() for c in self.components]
            self.reaction_name = "Campuran: " + " + ".join(names[:3])
            if len(names) > 3:
                self.reaction_name += "..."
            
            if self.current_pH < 4:
                self.description = f"Larutan asam kuat (pH {self.current_pH:.1f})"
            elif self.current_pH < 7:
                self.description = f"Larutan asam (pH {self.current_pH:.1f})"
            elif self.current_pH == 7:
                self.description = f"Larutan netral (pH {self.current_pH:.1f})"
            elif self.current_pH < 10:
                self.description = f"Larutan basa (pH {self.current_pH:.1f})"
            else:
                self.description = f"Larutan basa kuat (pH {self.current_pH:.1f})"
    
    def reset(self):
        """Reset campuran"""
        self.__init__()

# Global mixture instance
mixture = MixtureState()

# ============================================================================
# MEDIAPIPE INITIALIZATION
# ============================================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def apply_gaussian_smoothing(points, sigma=1.0):
    """Gaussian smoothing untuk cursor stabilization"""
    if len(points) < 2:
        return points[-1] if points else (0, 0)
    
    points_array = np.array(points)
    weights = np.exp(-np.arange(len(points))**2 / (2 * sigma**2))
    weights = weights / weights.sum()
    
    smoothed_x = np.average(points_array[:, 0], weights=weights)
    smoothed_y = np.average(points_array[:, 1], weights=weights)
    
    return (int(smoothed_x), int(smoothed_y))

def euclidean_distance(p1, p2):
    """Euclidean distance calculation"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def apply_edge_detection(roi):
    """Edge Detection untuk visualisasi"""
    if roi.size == 0:
        return roi
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

# ============================================================================
# GESTURE DETECTION
# ============================================================================

def detect_gesture(frame, hands_model):
    """Deteksi gesture menggunakan MediaPipe"""
    global cursor_history
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_model.process(frame_rgb)
    
    cursor_pos = None
    is_closed_fist = False
    hand_landmarks = None
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = hand_landmarks.landmark
        
        h, w, _ = frame.shape
        
        # Landmark 8: Ujung jari telunjuk
        index_tip = landmarks[8]
        cursor_x = int(index_tip.x * w)
        cursor_y = int(index_tip.y * h)
        
        # Landmark 4: Ujung jempol
        thumb_tip = landmarks[4]
        thumb_x = int(thumb_tip.x * w)
        thumb_y = int(thumb_tip.y * h)
        
        # Pinch detection
        pinch_distance = euclidean_distance(
            (cursor_x, cursor_y), 
            (thumb_x, thumb_y)
        )
        
        is_closed_fist = pinch_distance < 40
        
        # Smoothing cursor
        cursor_history.append((cursor_x, cursor_y))
        cursor_pos = apply_gaussian_smoothing(list(cursor_history))
        
    return cursor_pos, is_closed_fist, hand_landmarks

# ============================================================================
# PARTICLE SYSTEM
# ============================================================================

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = np.random.uniform(-2, 2)
        self.vy = np.random.uniform(-3, -1)
        self.color = color
        self.life = 255
        self.size = np.random.randint(3, 7)
    
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.2
        self.life -= 5
        
    def draw(self, frame):
        if self.life > 0:
            alpha = self.life / 255.0
            color = tuple([int(c * alpha) for c in self.color])
            cv2.circle(frame, (int(self.x), int(self.y)), self.size, color, -1)

def create_reaction_particles(center_x, center_y, color, count=20):
    """Buat particle effect"""
    global reaction_particles
    for _ in range(count):
        reaction_particles.append(Particle(center_x, center_y, color))

# ============================================================================
# AR ELEMENTS
# ============================================================================

def draw_button(frame, x, y, w, h, text, is_hovered, is_active=False):
    """Gambar tombol virtual"""
    if is_active:
        color = (100, 255, 100)
        thickness = 3
    elif is_hovered:
        color = (255, 200, 0)
        thickness = 2
    else:
        color = (255, 255, 255)
        thickness = 1
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    # Multi-line text untuk tombol kecil
    lines = text.split('\n')
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    line_height = 15
    
    total_height = len(lines) * line_height
    start_y = y + (h - total_height) // 2 + 12
    
    for i, line in enumerate(lines):
        text_size = cv2.getTextSize(line, font, font_scale, 1)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = start_y + i * line_height
        cv2.putText(frame, line, (text_x, text_y), font, font_scale, (255, 255, 255), 1)

def draw_reaction_vessel(frame, center_x, center_y):
    """Gambar wadah reaksi dengan liquid"""
    global mixture
    
    vessel_width = 180
    vessel_height = 220
    
    vessel_top_left = (center_x - vessel_width//2, center_y - vessel_height//2)
    vessel_bottom_right = (center_x + vessel_width//2, center_y + vessel_height//2)
    
    # Glass outline
    cv2.rectangle(frame, vessel_top_left, vessel_bottom_right, (200, 200, 200), 3)
    
    # Liquid level
    liquid_height = int(vessel_height * min(1.0, mixture.total_volume / 100))
    liquid_top = center_y + vessel_height//2 - liquid_height
    
    if liquid_height > 0:
        # Draw liquid
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                      (vessel_top_left[0], liquid_top),
                      (vessel_bottom_right[0], vessel_bottom_right[1]),
                      mixture.current_color, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Edge detection overlay
        roi_y1 = max(0, liquid_top)
        roi_y2 = min(frame.shape[0], vessel_bottom_right[1])
        roi_x1 = max(0, vessel_top_left[0])
        roi_x2 = min(frame.shape[1], vessel_bottom_right[0])
        
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
        if roi.size > 0:
            edges = apply_edge_detection(roi)
            edge_overlay = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            frame[roi_y1:roi_y2, roi_x1:roi_x2] = cv2.addWeighted(
                frame[roi_y1:roi_y2, roi_x1:roi_x2], 0.8, 
                edge_overlay, 0.2, 0
            )
    
    # Volume indicator
    cv2.putText(frame, f"Volume: {mixture.total_volume} mL", 
                (center_x - 70, center_y + vessel_height//2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_ar_elements(frame, cursor_pos, is_clicking):
    """Render semua elemen AR"""
    global mixture, last_click_time, reaction_particles
    
    h, w, _ = frame.shape
    
    # Define chemical buttons (2 kolom)
    buttons = []
    chemicals_list = list(CHEMICALS_DATABASE.keys())
    
    col1_x, col2_x = 20, 180
    start_y = 20
    button_w, button_h = 140, 55
    gap = 10
    
    for idx, chem_id in enumerate(chemicals_list):
        row = idx // 2
        col = idx % 2
        x = col1_x if col == 0 else col2_x
        y = start_y + row * (button_h + gap)
        
        chem = CHEMICALS_DATABASE[chem_id]
        button_text = f"{chem_id}\n{chem['description']}"
        
        buttons.append({
            "rect": (x, y, button_w, button_h),
            "text": button_text,
            "action": chem_id,
            "color": chem["color"]
        })
    
    # Reset button
    reset_y = start_y + ((len(chemicals_list) + 1) // 2) * (button_h + gap)
    buttons.append({
        "rect": (col1_x, reset_y, button_w * 2 + col2_x - col1_x, 50),
        "text": "RESET WADAH",
        "action": "RESET",
        "color": (0, 0, 255)
    })
    
    # Check interaction
    current_time = time.time()
    can_click = (current_time - last_click_time) > click_cooldown
    
    for btn in buttons:
        x, y, bw, bh = btn["rect"]
        is_hovered = False
        
        if cursor_pos:
            cx, cy = cursor_pos
            if x <= cx <= x + bw and y <= cy <= y + bh:
                is_hovered = True
                
                if is_clicking and can_click:
                    action = btn["action"]
                    
                    if action == "RESET":
                        mixture.reset()
                        reaction_particles.clear()
                    else:
                        mixture.add_chemical(action, volume=10)
                        create_reaction_particles(600, 380, btn["color"], 15)
                    
                    last_click_time = current_time
        
        draw_button(frame, x, y, bw, bh, btn["text"], is_hovered)
    
    # Draw vessel
    draw_reaction_vessel(frame, 600, 380)
    
    # Draw particles
    for particle in reaction_particles[:]:
        particle.update()
        particle.draw(frame)
        if particle.life <= 0:
            reaction_particles.remove(particle)
    
    # pH Bar visualization
    ph_bar_x, ph_bar_y = 450, 200
    ph_bar_w, ph_bar_h = 300, 30
    
    # pH gradient background
    for i in range(ph_bar_w):
        pH_val = (i / ph_bar_w) * 14
        if pH_val < 7:
            r = int(255 * (7 - pH_val) / 7)
            g = int(255 * pH_val / 7)
            b = 0
        else:
            r = 0
            g = int(255 * (14 - pH_val) / 7)
            b = int(255 * (pH_val - 7) / 7)
        
        cv2.line(frame, (ph_bar_x + i, ph_bar_y), 
                 (ph_bar_x + i, ph_bar_y + ph_bar_h), (b, g, r), 1)
    
    cv2.rectangle(frame, (ph_bar_x, ph_bar_y), 
                  (ph_bar_x + ph_bar_w, ph_bar_y + ph_bar_h), (255, 255, 255), 2)
    
    # pH indicator
    ph_pos_x = ph_bar_x + int((mixture.current_pH / 14) * ph_bar_w)
    cv2.line(frame, (ph_pos_x, ph_bar_y - 10), (ph_pos_x, ph_bar_y + ph_bar_h + 10), 
             (0, 255, 255), 3)
    
    # pH labels
    cv2.putText(frame, "0", (ph_bar_x - 10, ph_bar_y + ph_bar_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "7", (ph_bar_x + ph_bar_w//2 - 5, ph_bar_y + ph_bar_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "14", (ph_bar_x + ph_bar_w - 10, ph_bar_y + ph_bar_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Status panel
    status_x, status_y = 420, 520
    status_w, status_h = 400, 150
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (status_x, status_y), 
                  (status_x + status_w, status_y + status_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.rectangle(frame, (status_x, status_y), 
                  (status_x + status_w, status_y + status_h), (255, 255, 255), 2)
    
    # Status text
    cv2.putText(frame, "HASIL CAMPURAN:", (status_x + 10, status_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.putText(frame, mixture.reaction_name, (status_x + 10, status_y + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    
    cv2.putText(frame, mixture.description, (status_x + 10, status_y + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.putText(frame, f"pH Aktual: {mixture.current_pH:.2f}", (status_x + 10, status_y + 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    
    # Component list
    if mixture.components:
        cv2.putText(frame, "Komponen:", (status_x + 10, status_y + 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        comp_text = ", ".join([f"{c[0]} ({c[1]}mL)" for c in mixture.components[:4]])
        if len(mixture.components) > 4:
            comp_text += "..."
        cv2.putText(frame, comp_text, (status_x + 80, status_y + 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    
    # Draw cursor
    if cursor_pos:
        color = (0, 255, 0) if is_clicking else (255, 255, 0)
        cv2.circle(frame, cursor_pos, 15, color, 2)
        cv2.circle(frame, cursor_pos, 3, color, -1)

# ============================================================================
# MAIN PROGRAM (SUDAH DIPERBAIKI)
# ============================================================================

def main():
    """Main loop"""
    global hands
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    print("="*60)
    print("  CHEM-AQUA LAB - Virtual Chemistry Laboratory")
    print("="*60)
    print("\n📌 KONTROL GESTURE:")
    print("  • Gerakkan jari telunjuk untuk kontrol cursor")
    print("  • Cubit (telunjuk + jempol) untuk klik tombol")
    print("  • Klik bahan kimia untuk menambahkan ke wadah")
    print("  • Klik RESET untuk mengosongkan wadah")
    print("\n🧪 BAHAN TERSEDIA:")
    for chem_id, chem in CHEMICALS_DATABASE.items():
        print(f"  • {chem_id}: {chem['name']} (pH {chem['pH']})")
    print("\n⌨️  Tekan 'q' untuk keluar")
    print("="*60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # >>>>>> BARIS PERBAIKAN DITAMBAHKAN DI SINI <<<<<<
        h, w, _ = frame.shape
        
        cursor_pos, is_clicking, hand_landmarks = detect_gesture(frame, hands)
        
        if hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255,255,255), thickness=1)
            )
        
        draw_ar_elements(frame, cursor_pos, is_clicking)
        
        # Instructions (Sekarang variabel 'h' sudah terdefinisi)
        cv2.putText(frame, "Cubit untuk klik | 'Q' = Keluar", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("CHEM-AQUA LAB - Virtual Chemistry Lab", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\n✅ Program terminated successfully")

if __name__ == "__main__":
    main()