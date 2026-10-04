import base64
import io
import unittest

from PIL import Image, ImageDraw

from mj_env.image_regions import color_regions, track_region


class ImageRegionTests(unittest.TestCase):
    def encoded(self, box):
        image = Image.new('RGB', (640, 480), (240, 220, 190))
        ImageDraw.Draw(image).rectangle(box, fill='blue')
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()

    def test_each_camera_uses_its_own_original_pixels(self):
        sample = {'images_png_base64': {
            'global': self.encoded((20, 30, 60, 70)),
            'wrist': self.encoded((300, 200, 340, 240))}}
        for camera, center in [('global', [40, 50]), ('wrist', [320, 220])]:
            result = color_regions(sample, camera)
            self.assertEqual(result['camera'], camera)
            self.assertEqual(result['image_size'], [640, 480])
            blue = [r for r in result['regions'] if r['color'] == 'blue']
            self.assertEqual(len(blue), 1)
            self.assertEqual(blue[0]['center_px'], center)
            self.assertNotIn('position_m', blue[0])

    def test_missing_camera_does_not_substitute_other_camera(self):
        sample = {'images_png_base64': {'global': self.encoded((20, 30, 60, 70))}}
        self.assertEqual(color_regions(sample, 'wrist')['status'], 'missing image')
        with self.assertRaises(ValueError):
            color_regions(sample, 'depth')

    def test_track_does_not_jump_to_distant_same_color_object(self):
        cube=dict(color='blue',center_px=[180,300],area_px=1000)
        sphere=dict(color='blue',center_px=[510,250],area_px=950)
        reference=dict(region=cube,status='selected')
        lost=track_region(reference,{'regions':[sphere]})
        self.assertEqual(lost['status'],'lost')
        self.assertEqual(track_region(lost,{'regions':[cube]})['status'],'lost')
        moved=dict(cube,center_px=[185,302],area_px=1100)
        tracked=track_region(reference,{'regions':[sphere,moved]})
        self.assertEqual(tracked['delta_px'],[5,2])
        self.assertEqual(tracked['area_ratio'],1.1)
        self.assertEqual(tracked['region'],moved)

    def test_ambiguous_components_are_not_silently_chosen(self):
        region=dict(color='blue',center_px=[100,100],area_px=1000)
        result=track_region(dict(region=region),{'regions':[
            dict(region,center_px=[90,100]),dict(region,center_px=[110,100])]})
        self.assertEqual(result['status'],'lost')


if __name__ == '__main__':
    unittest.main()
