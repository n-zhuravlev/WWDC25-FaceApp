//
//  ViewController.swift
//  ml_init_ios18
//
//  Created by Evgenii Bogomolov on 10/06/2025.
//

import UIKit
import CoreML

class ViewController: UIViewController {
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        Task(priority: .userInitiated) {
            await Test.testInitialization(config: .cpuAndNeuralEngine)
        }
    }
}

enum Test {
    
    static func testInitialization(config: MLModelConfiguration) async {
        let version = await UIDevice.current.deviceInfo()
        do {
            print("start: \(config.computeUnits), device: \(version)")
            let start = Date()
            let model = try problem1(configuration: config)
            print("init time: \(Date().timeIntervalSince(start)) seconds")
            
            let size = CGSize(width: 1024, height: 1024)
            testPrediction(model: model, image: UIImage.make(size: size))
            testPrediction(model: model, image: UIImage.make(size: size))
        } catch {
            print("init failed: \(error)")
        }
    }
    
    static func testPrediction(model: problem1, image: UIImage) {
        do {
            let input = try problem1Input(input_imageWith: image.cgImage!)
            let start = Date()
            let output = try model.prediction(input: input)
            print("prediction time: \(Date().timeIntervalSince(start)) seconds")
        } catch {
            print("prediction failed: \(error)")
        }
    }
}

extension UIDevice {
    func deviceInfo() -> String {
        var systemInfo = utsname()
        uname(&systemInfo)
        let utf8String = NSString(
            bytes: &systemInfo.machine,
            length: Int(_SYS_NAMELEN),
            encoding: String.Encoding.ascii.rawValue
        )!.utf8String!
        let versionCode: String = String(validatingUTF8: utf8String) ?? "Unknown"
        return versionCode + " " + self.systemName + " " + self.systemVersion
    }
}

extension MLModelConfiguration {
    static var cpuAndGPU: MLModelConfiguration {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU
        return config
    }
    
    static var cpuAndNeuralEngine: MLModelConfiguration {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        return config
    }
    
    static var all: MLModelConfiguration {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        return config
    }
}

extension MLComputeUnits: @retroactive CustomDebugStringConvertible {
    public var debugDescription: String {
        switch self {
        case .cpuOnly:
            return "CPU only"
        case .cpuAndGPU:
            return "CPU and GPU"
        case .cpuAndNeuralEngine:
            return "CPU and Neural Engine"
        case .all:
            return "All available compute units"
        @unknown default:
            return "unknown"
        }
    }
}

extension UIImage {
    
    static func make(size: CGSize) -> UIImage {
        let renderer = UIGraphicsImageRenderer(bounds: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        return renderer.image { _ in }
    }
}

extension CGImage {
    static func makeMask(size: CGSize) -> CGImage {
        let width = Int(size.width)
        let height = Int(size.height)
        var info = CGBitmapInfo()
        info.formUnion(CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue))
        info.formUnion(CGBitmapInfo.byteOrderDefault)
        info.formUnion(CGBitmapInfo(rawValue: CGImagePixelFormatInfo.packed.rawValue))
        let data = Data(count: width * height * 8)
        let provider = CGDataProvider(data: data as CFData)!
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            bytesPerRow: width * 8,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: info,
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )!
    }
}

extension FileManager {
    func printFilesAndSizes(in directoryURL: URL, padding: Int = 0) {
        let fileManager = self
        let pad = String(repeating: "   ", count: padding)
        do {
            let resourceKeys: [URLResourceKey] = [.fileSizeKey, .isDirectoryKey]
            let fileURLs = try fileManager.contentsOfDirectory(at: directoryURL, includingPropertiesForKeys: resourceKeys, options: [])
            
            for fileURL in fileURLs {
                let resourceValues = try fileURL.resourceValues(forKeys: Set(resourceKeys))
                
                if resourceValues.isDirectory == true {
                    print("\(pad)📁 Directory: \(fileURL.lastPathComponent)")
                    self.printFilesAndSizes(in: fileURL, padding: padding + 1)
                } else {
                    let size = resourceValues.fileSize ?? 0
                    print("\(pad)📄 File: \(fileURL.lastPathComponent) – \(size) bytes")
                }
            }
        } catch {
            print("Error reading directory contents: \(error)")
        }
    }
}
