#!/usr/bin/env python3
"""Test script to verify XMP preservation functionality"""

import re
import tempfile
from pathlib import Path


def has_existing_crop(xmp_path: Path):
    """Check if XMP file exists and already has crop data"""
    if not xmp_path.exists():
        return False
    
    try:
        content = xmp_path.read_text()
        # Look for HasCrop tag with True value
        return bool(re.search(r'<crs:HasCrop>\s*(True|true|1)\s*</crs:HasCrop>', content))
    except Exception:
        return False


def write_xmp(xmp_path: Path, left: float, top: float, right: float, bottom: float):
    """Simplified version of write_xmp for testing"""
    
    crop_tags = {
        'HasCrop': 'True',
        'CropLeft': f'{left:.6f}',
        'CropTop': f'{top:.6f}',
        'CropRight': f'{right:.6f}',
        'CropBottom': f'{bottom:.6f}'
    }

    if xmp_path.exists():
        # Read existing XMP and update crop tags using regex
        content = xmp_path.read_text()
        
        for tag, value in crop_tags.items():
            # Pattern to match existing tag with any content
            pattern = rf'<crs:{tag}>.*?</crs:{tag}>'
            replacement = f'<crs:{tag}>{value}</crs:{tag}>'
            
            if re.search(pattern, content):
                # Tag exists, replace it
                content = re.sub(pattern, replacement, content)
            else:
                # Tag doesn't exist, insert it before the closing Description tag
                desc_close = '</rdf:Description>'
                if desc_close in content:
                    # Insert new tag before closing Description
                    new_tag = f'   <crs:{tag}>{value}</crs:{tag}>\n  '
                    content = content.replace(desc_close, new_tag + desc_close)
        
        xmp_path.write_text(content)

    else:
        # Create new XMP file with crop data
        xmp = f"""<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">
   <crs:HasCrop>True</crs:HasCrop>
   <crs:CropLeft>{left:.6f}</crs:CropLeft>
   <crs:CropTop>{top:.6f}</crs:CropTop>
   <crs:CropRight>{right:.6f}</crs:CropRight>
   <crs:CropBottom>{bottom:.6f}</crs:CropBottom>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""

        xmp_path.write_text(xmp)


def test_new_xmp():
    """Test creating a new XMP file"""
    print("\n=== Test 1: Creating new XMP file ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        xmp_path = Path(tmpdir) / "test.xmp"
        write_xmp(xmp_path, 0.1, 0.2, 0.9, 0.8)
        
        content = xmp_path.read_text()
        print("✓ New XMP file created")
        assert "CropLeft>0.100000" in content
        assert "CropTop>0.200000" in content
        print("✓ Crop values are correct")


def test_existing_xmp_with_other_metadata():
    """Test preserving existing XMP metadata while updating crop"""
    print("\n=== Test 2: Preserving existing metadata ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        xmp_path = Path(tmpdir) / "test.xmp"
        
        # Create an XMP with existing metadata
        existing_xmp = """<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
    xmlns:exif="http://ns.adobe.com/exif/1.0/">
   <crs:Temperature>5500</crs:Temperature>
   <crs:Exposure>+0.50</crs:Exposure>
   <exif:ISO>800</exif:ISO>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""
        
        xmp_path.write_text(existing_xmp)
        print("✓ Created XMP with existing metadata")
        
        # Update with crop data
        write_xmp(xmp_path, 0.15, 0.25, 0.85, 0.75)
        
        content = xmp_path.read_text()
        print("✓ Updated XMP with crop data")
        
        # Check that crop data was added
        assert "CropLeft>0.150000" in content
        assert "CropTop>0.250000" in content
        print("✓ Crop values are correct")
        
        # Check that original metadata is preserved
        assert "Temperature>5500" in content
        assert "Exposure>+0.50" in content
        assert "ISO>800" in content
        print("✓ Original metadata preserved")


def test_updating_existing_crop():
    """Test updating existing crop values"""
    print("\n=== Test 3: Updating existing crop values ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        xmp_path = Path(tmpdir) / "test.xmp"
        
        # Create initial crop
        write_xmp(xmp_path, 0.1, 0.2, 0.9, 0.8)
        print("✓ Created initial crop")
        
        # Update crop
        write_xmp(xmp_path, 0.3, 0.4, 0.7, 0.6)
        
        content = xmp_path.read_text()
        print("✓ Updated crop values")
        
        # Check new values are present
        assert "CropLeft>0.300000" in content
        assert "CropTop>0.400000" in content
        assert "CropRight>0.700000" in content
        assert "CropBottom>0.600000" in content
        print("✓ New crop values are correct")
        
        # Check old values are NOT present
        assert "CropLeft>0.100000" not in content
        assert "CropTop>0.200000" not in content
        print("✓ Old crop values removed")


def test_has_existing_crop():
    """Test the has_existing_crop detection function"""
    print("\n=== Test 4: Detecting existing crops ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        xmp_path = Path(tmpdir) / "test.xmp"
        
        # Test 1: No XMP file
        assert not has_existing_crop(xmp_path)
        print("✓ Returns False when XMP doesn't exist")
        
        # Test 2: XMP without crop
        xmp_no_crop = """<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">
   <crs:Temperature>5500</crs:Temperature>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""
        xmp_path.write_text(xmp_no_crop)
        assert not has_existing_crop(xmp_path)
        print("✓ Returns False when XMP exists but has no crop")
        
        # Test 3: XMP with crop
        write_xmp(xmp_path, 0.1, 0.2, 0.9, 0.8)
        assert has_existing_crop(xmp_path)
        print("✓ Returns True when XMP has crop data")


def main():
    print("Testing XMP Preservation Functionality")
    print("=" * 50)
    
    try:
        test_new_xmp()
        test_existing_xmp_with_other_metadata()
        test_updating_existing_crop()
        test_has_existing_crop()
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("=" * 50)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
