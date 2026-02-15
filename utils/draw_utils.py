import cv2
import numpy as np

class CameraVisualizer:
    def __init__(self, width=1280, height=720, vis_scale=10.0):
        self.width = width
        self.height = height
        self.vis_scale = vis_scale
        
        f = max(width, height) * 0.7
        self.K = np.array([
            [f, 0, width // 2],
            [0, f, height // 2],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros(4)

        s = 0.1
        self.camera_model = np.array([
            [0, 0, 0], [s, s, s*2], [-s, s, s*2], [-s, -s, s*2], [s, -s, s*2]
        ], dtype=np.float32)
        
        self.camera_map = {}

    def _scale_pose(self, T_cw):
        T = T_cw.copy()
        T[:3, 3] *= self.vis_scale
        return T

    def visualize(self, pose, current_frame):
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8) + 15
        
        self.camera_map[current_frame] = pose 

        T_cw_curr = self._scale_pose(self.camera_map[current_frame])
        R_curr = T_cw_curr[:3, :3]
        t_curr = T_cw_curr[:3, 3]
        
        follow_dist = 6.0
        up_offset = 0.5
        
        z_vec = R_curr[:, 2]
        y_vec = R_curr[:, 1]
        viewer_pos = t_curr - z_vec * follow_dist - y_vec * up_offset
        
        T_vw = np.eye(4)
        T_vw[:3, :3] = R_curr
        T_vw[:3, 3] = viewer_pos
        
        T_wv = np.linalg.inv(T_vw)
        rvec, _ = cv2.Rodrigues(T_wv[:3, :3])
        tvec = T_wv[:3, 3]

        history_centers = np.array(
            [self._scale_pose(p)[:3, 3] 
            for i, p in self.camera_map.items() if i < current_frame],
            dtype=np.float32
        )
        
        if len(history_centers) > 0:
            img_pts, _ = cv2.projectPoints(history_centers, rvec, tvec, self.K, self.dist_coeffs)
            img_pts = img_pts.reshape(-1, 2)
            
            for i in range(len(img_pts)):
                p_world = np.append(history_centers[i], 1.0)
                p_viewer = T_wv @ p_world
                
                if p_viewer[2] > 0.5: 
                    u, v = img_pts[i].astype(int)
                    if 0 <= u < self.width and 0 <= v < self.height:
                        color_ratio = i / (current_frame + 1)
                        color = (int(255 * (1 - color_ratio)), int(255 * color_ratio), 200)
                        
                        radius = max(2, int(8 / (p_viewer[2] * 0.2 + 1)))
                        cv2.circle(canvas, (u, v), radius, color, -1, cv2.LINE_AA)

        # self._draw_camera_frustum(canvas, T_cw_curr, rvec, tvec, color=(0, 255, 255))

        cv2.rectangle(canvas, (0, 0), (self.width, 60), (40, 40, 40), -1)
        cv2.putText(canvas, f"TRACKING FRAME: {current_frame}", (30, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return canvas

    # def _draw_camera_frustum(self, canvas, T_cw, rvec_v, tvec_v, color):
    #     R, t = T_cw[:3, :3], T_cw[:3, 3]
    #     pts_w = (R @ self.camera_model.T).T + t
    #     img_pts, _ = cv2.projectPoints(pts_w, rvec_v, tvec_v, self.K, self.dist_coeffs)
    #     img_pts = img_pts.reshape(-1, 2).astype(int)
    #     print(img_pts)
    #     center = tuple(img_pts[0])
    #     for i in range(1, 5):
    #         cv2.line(canvas, center, tuple(img_pts[i]), color, 3, cv2.LINE_AA)
    #     for i in range(1, 5):
    #         next_i = i + 1 if i < 4 else 1
    #         cv2.line(canvas, tuple(img_pts[i]), tuple(img_pts[next_i]), color, 3, cv2.LINE_AA)