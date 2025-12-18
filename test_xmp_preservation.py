#!/usr/bin/env python3
"""Test script to verify XMP preservation functionality"""

import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

# Simulate the write_xmp function
def write_xmp(xmp_path: Path, left: float, top: float, right: float, bottom: float):
    """Simplified version of write_xmp for testing"""
    
    # Define namespaces
    namespaces = {
        'x': 'adobe:ns:meta/',
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'crs': 'http://ns.adobe.com/camera-raw-settings/1.0/'
    }

    # Register namespaces for proper serialization
    for prefix, uri in namespaces.items():
        ET.register_namespace(prefix, uri)

    if xmp_path.exists():
        # Read and parse existing XMP
        tree = ET.parse(xmp_path)
        root = tree.getroot()

        # Find or create the RDF Description element with crs namespace
        rdf = root.find('.//rdf:RDF', namespaces)
        if rdf is None:
            # Create RDF structure if it doesn't exist
            rdf = ET.SubElement(root, f"{{{namespaces['rdf']}}}RDF")
        
        # Find first Description element (there may be multiple)
        desc = rdf.find('.//rdf:Description', namespaces)
        if desc is None:
            # Create new Description element
            desc = ET.SubElement(rdf, f"{{{namespaces['rdf']}}}Description")
            desc.set(f"{{{namespaces['rdf']}}}about", "")
        
        # Ensure crs namespace is declared on Description element
        desc.set(f"{{http://www.w3.org/2000/xmlns/}}crs", namespaces['crs'])

        # Update or create crop tags
        crop_tags = {
            f"{{{namespaces['crs']}}}HasCrop": "True",
            f"{{{namespaces['crs']}}}CropLeft": f"{left:.6f}",
            f"{{{namespaces['crs']}}}CropTop": f"{top:.6f}",
            f"{{{namespaces['crs']}}}CropRight": f"{right:.6f}",
            f"{{{namespaces['crs']}}}CropBottom": f"{bottom:.6f}"
        }

        for tag, value in crop_tags.items():
            elem = desc.find(f".//{tag}", namespaces)
            if elem is not None:
                desc.remove(elem)
            new_elem = ET.SubElement(desc, tag)
            new_elem.text = value

        # Write back with XML declaration and xpacket wrapper
        tree.write(xmp_path, encoding='utf-8', xml_declaration=True)
        
        # Add xpacket processing instructions
        content = xmp_path.read_text()
        if not content.startswith('<?xpacket'):
            content = f'<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>\n{content}'
        if not content.endswith('<?xpacket end="w"?>'):
            content = f'{content.rstrip()}\n<?xpacket end="w"?>'
        xmp_path.write_text(content)

    else:
        # Create new XMP file with crop data (original behavior)
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


def main():
    print("Testing XMP Preservation Functionality")
    print("=" * 50)
    
    try:
        test_new_xmp()
        test_existing_xmp_with_other_metadata()
        test_updating_existing_crop()
        
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
