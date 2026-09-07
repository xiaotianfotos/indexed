// SPDX-License-Identifier: Apache-2.0
// Private ANE selector, blob and IOSurface conventions adapted from oMLX v0.6.4,
// commit 1d7826185c5b5b69b38b27cbe57d7597b7551fd7, qwen35_ane.mm.
// Indexed modifications: narrow C ABI; no Python/MLX/Metal dependency; bounded
// copies; serialized synchronous evaluation; exclusive staging ownership.
// See Vendor/omlx-ane/README.md and LICENSE for attribution.
#import "PrivateANEBridge.h"
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

static void fail(char *buffer, size_t capacity, NSString *message) {
    if (buffer && capacity) snprintf(buffer, capacity, "%s", message.UTF8String ?: "Private ANE failure");
}

@interface IndexedANEState : NSObject
@property(nonatomic, strong) id model;
@property(nonatomic, strong) id request;
@property(nonatomic, strong) NSMutableArray *inputs;
@property(nonatomic, strong) NSMutableArray *outputs;
@property(nonatomic, strong) NSArray<NSNumber *> *inputSizes;
@property(nonatomic, strong) NSArray<NSNumber *> *outputSizes;
@property(nonatomic, copy) NSString *directory;
@property(nonatomic) int lockFD;
@property(nonatomic) BOOL loaded;
@property(nonatomic) BOOL failed;
@end

@implementation IndexedANEState
- (instancetype)init {
    if ((self = [super init])) { _lockFD = -1; _inputs = [NSMutableArray new]; _outputs = [NSMutableArray new]; }
    return self;
}
- (void)dealloc {
    if (_loaded && [_model respondsToSelector:@selector(unloadWithQoS:error:)]) {
        NSError *error = nil;
        ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(_model, @selector(unloadWithQoS:error:), 21, &error);
    }
    if (_directory) [[NSFileManager defaultManager] removeItemAtPath:_directory error:nil];
    if (_lockFD >= 0) { flock(_lockFD, LOCK_UN); close(_lockFD); }
}
@end

static IndexedANEState *state(IndexedANEProgram *program) { return (__bridge IndexedANEState *)program; }

static NSData *blob(const uint8_t *bytes, size_t count) {
    NSMutableData *data = [NSMutableData dataWithLength:128 + count];
    uint8_t *target = data.mutableBytes;
    target[0] = 1; target[4] = 2;
    uint32_t magic = 0xdeadbeef, type = 1;
    uint64_t length = count, offset = 128;
    memcpy(target + 64, &magic, 4); memcpy(target + 68, &type, 4);
    memcpy(target + 72, &length, 8); memcpy(target + 80, &offset, 8);
    memcpy(target + 128, bytes, count);
    return data;
}

static IOSurfaceRef surface(size_t bytes) {
    size_t allocation = MAX((size_t)65536, (bytes + 65535) & ~(size_t)65535);
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(allocation), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(allocation),
        (id)kIOSurfaceAllocSize: @(allocation), (id)kIOSurfacePixelFormat: @0
    });
}

IndexedANEProgram *indexed_ane_create_sharing(
    const uint8_t *mil, size_t mil_bytes,
    const uint8_t *weight_data, size_t weight_data_bytes,
    const uint8_t *weight_scales, size_t weight_scales_bytes,
    const size_t *input_sizes, size_t input_count,
    const size_t *output_sizes, size_t output_count,
    IndexedANEProgram *sharing,
    char *error, size_t error_capacity) {
    @autoreleasepool { @try {
        if (!mil || !mil_bytes || mil_bytes > 16 * 1024 * 1024 || !input_sizes || !output_sizes
            || !input_count || input_count > 32 || !output_count || output_count > 32
            || weight_data_bytes > 1024ULL * 1024 * 1024 || weight_scales_bytes > 16 * 1024 * 1024
            || (weight_data_bytes && !weight_data) || (weight_scales_bytes && !weight_scales)) {
            fail(error, error_capacity, @"Invalid private ANE graph or buffer limits"); return NULL;
        }
        size_t total = 0;
        for (size_t group = 0; group < 2; group++) {
            const size_t *sizes = group ? output_sizes : input_sizes;
            size_t count = group ? output_count : input_count;
            for (size_t i = 0; i < count; i++) {
                if (!sizes[i] || sizes[i] > 256 * 1024 * 1024) { fail(error, error_capacity, @"Invalid private ANE surface size"); return NULL; }
                total += sizes[i];
            }
        }
        if (total > 512 * 1024 * 1024) { fail(error, error_capacity, @"Private ANE surfaces exceed the memory budget"); return NULL; }
        IndexedANEState *shared = state(sharing);
        if (shared) {
            if (shared.inputSizes.count != input_count || shared.outputSizes.count != output_count) {
                fail(error, error_capacity, @"Shared private ANE surface count differs"); return NULL;
            }
            for (size_t i = 0; i < input_count; i++) if (shared.inputSizes[i].unsignedLongLongValue != input_sizes[i]) {
                fail(error, error_capacity, @"Shared private ANE input size differs"); return NULL;
            }
            for (size_t i = 0; i < output_count; i++) if (shared.outputSizes[i].unsignedLongLongValue != output_sizes[i]) {
                fail(error, error_capacity, @"Shared private ANE output size differs"); return NULL;
            }
        }
        static void *framework;
        static dispatch_once_t once;
        dispatch_once(&once, ^{ framework = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW | RTLD_LOCAL); });
        Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class modelClass = NSClassFromString(@"_ANEInMemoryModel");
        Class requestClass = NSClassFromString(@"_ANERequest");
        Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
        if (!framework || !descriptorClass || !modelClass || !requestClass || !surfaceClass) { fail(error, error_capacity, @"Private ANE runtime unavailable"); return NULL; }
        NSData *text = [NSData dataWithBytes:mil length:mil_bytes];
        NSMutableDictionary *weights = [NSMutableDictionary new], *files = [NSMutableDictionary new];
        if (weight_data_bytes) files[@"weight_data.bin"] = blob(weight_data, weight_data_bytes);
        if (weight_scales_bytes) files[@"weight_scale.bin"] = blob(weight_scales, weight_scales_bytes);
        for (NSString *name in files) weights[[@"@model_path/weights/" stringByAppendingString:name]] = @{@"offset": @0, @"data": files[name]};
        id descriptor = ((id (*)(id, SEL, id, id, id))objc_msgSend)(descriptorClass, @selector(modelWithMILText:weights:optionsPlist:), text, weights, nil);
        IndexedANEState *owner = [IndexedANEState new];
        owner.model = ((id (*)(id, SEL, id))objc_msgSend)(modelClass, @selector(inMemoryModelWithDescriptor:), descriptor);
        if (!owner.model) { fail(error, error_capacity, @"Private ANE descriptor creation failed"); return NULL; }
        NSString *identifier = ((id (*)(id, SEL))objc_msgSend)(owner.model, @selector(hexStringIdentifier));
        if (![identifier isKindOfClass:NSString.class] || !identifier.length || identifier.length > 240 || [identifier rangeOfCharacterFromSet:[[NSCharacterSet characterSetWithCharactersInString:@"0123456789abcdefABCDEF_"] invertedSet]].location != NSNotFound) {
            fail(error, error_capacity, @"Invalid private ANE descriptor identity"); return NULL;
        }
        NSString *directory = [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
        NSString *lockPath = [NSTemporaryDirectory() stringByAppendingPathComponent:[@"indexed-ane-" stringByAppendingString:identifier]];
        owner.lockFD = open(lockPath.fileSystemRepresentation, O_CREAT | O_RDWR | O_NOFOLLOW, 0600);
        if (owner.lockFD < 0 || flock(owner.lockFD, LOCK_EX | LOCK_NB) != 0) { fail(error, error_capacity, @"Private ANE program is already owned by another process"); return NULL; }
        NSFileManager *manager = NSFileManager.defaultManager;
        // Do not delete a pre-existing staging directory owned by another implementation.
        if ([manager fileExistsAtPath:directory]) { fail(error, error_capacity, @"Private ANE staging directory already exists"); return NULL; }
        NSError *nativeError = nil;
        if (![manager createDirectoryAtPath:[directory stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:@{NSFilePosixPermissions: @0700} error:&nativeError]) {
            fail(error, error_capacity, nativeError.localizedDescription); return NULL;
        }
        owner.directory = directory;
        if (![text writeToFile:[directory stringByAppendingPathComponent:@"model.mil"] options:NSDataWritingAtomic error:&nativeError]) { fail(error, error_capacity, nativeError.localizedDescription); return NULL; }
        for (NSString *name in files) if (![files[name] writeToFile:[[directory stringByAppendingPathComponent:@"weights"] stringByAppendingPathComponent:name] options:NSDataWritingAtomic error:&nativeError]) { fail(error, error_capacity, nativeError.localizedDescription); return NULL; }
        if (!((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(owner.model, @selector(compileWithQoS:options:error:), 21, @{}, &nativeError)
            || !((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(owner.model, @selector(loadWithQoS:options:error:), 21, @{}, &nativeError)) {
            fail(error, error_capacity, nativeError.localizedDescription ?: @"Private ANE compile/load failed"); return NULL;
        }
        owner.loaded = YES;
        NSMutableArray *inObjects = [NSMutableArray new], *outObjects = [NSMutableArray new], *inIndices = [NSMutableArray new], *outIndices = [NSMutableArray new];
        NSMutableArray *inSizes = [NSMutableArray new], *outSizes = [NSMutableArray new];
        for (size_t group = 0; group < 2; group++) {
            size_t count = group ? output_count : input_count;
            const size_t *sizes = group ? output_sizes : input_sizes;
            for (size_t i = 0; i < count; i++) {
                IOSurfaceRef value = shared
                    ? (IOSurfaceRef)CFRetain((__bridge CFTypeRef)(group ? shared.outputs[i] : shared.inputs[i]))
                    : surface(sizes[i]);
                if (!value) { fail(error, error_capacity, @"Private ANE IOSurface allocation failed"); return NULL; }
                id wrapped = ((id (*)(id, SEL, IOSurfaceRef))objc_msgSend)(surfaceClass, @selector(objectWithIOSurface:), value);
                [(group ? owner.outputs : owner.inputs) addObject:CFBridgingRelease(value)];
                if (!wrapped) { fail(error, error_capacity, @"Private ANE surface wrapper failed"); return NULL; }
                [(group ? outObjects : inObjects) addObject:wrapped];
                [(group ? outIndices : inIndices) addObject:@(i)];
                [(group ? outSizes : inSizes) addObject:@(sizes[i])];
            }
        }
        owner.inputSizes = inSizes; owner.outputSizes = outSizes;
        owner.request = ((id (*)(id, SEL, id, id, id, id, id, id, id))objc_msgSend)(requestClass,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:), inObjects, inIndices, outObjects, outIndices, nil, nil, @0);
        if (!owner.request) { fail(error, error_capacity, @"Private ANE request creation failed"); return NULL; }
        return (IndexedANEProgram *)CFBridgingRetain(owner);
    } @catch (NSException *exception) { fail(error, error_capacity, exception.reason); return NULL; } }
}

IndexedANEProgram *indexed_ane_create(
    const uint8_t *mil, size_t mil_bytes,
    const uint8_t *weight_data, size_t weight_data_bytes,
    const uint8_t *weight_scales, size_t weight_scales_bytes,
    const size_t *input_sizes, size_t input_count,
    const size_t *output_sizes, size_t output_count,
    char *error, size_t error_capacity) {
    return indexed_ane_create_sharing(mil, mil_bytes, weight_data, weight_data_bytes,
        weight_scales, weight_scales_bytes, input_sizes, input_count,
        output_sizes, output_count, NULL, error, error_capacity);
}

bool indexed_ane_write(IndexedANEProgram *program, size_t index, const void *bytes, size_t count, char *error, size_t error_capacity) {
    IndexedANEState *owner = state(program);
    if (!owner || !bytes || index >= owner.inputs.count || count != owner.inputSizes[index].unsignedLongLongValue) { fail(error, error_capacity, @"Invalid private ANE input bounds"); return false; }
    IOSurfaceRef value = (__bridge IOSurfaceRef)owner.inputs[index];
    if (IOSurfaceLock(value, 0, NULL) != kIOReturnSuccess) { fail(error, error_capacity, @"Private ANE input lock failed"); return false; }
    memcpy(IOSurfaceGetBaseAddress(value), bytes, count);
    IOSurfaceUnlock(value, 0, NULL);
    return true;
}

bool indexed_ane_evaluate(IndexedANEProgram *program, char *error, size_t error_capacity) {
    @autoreleasepool { @try {
        IndexedANEState *owner = state(program);
        if (!owner || owner.failed) { fail(error, error_capacity, @"Private ANE program unavailable after failure"); return false; }
        NSError *nativeError = nil;
        BOOL result = ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(owner.model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, owner.request, &nativeError);
        if (!result) { owner.failed = YES; fail(error, error_capacity, nativeError.localizedDescription ?: @"Private ANE evaluation failed"); }
        return result;
    } @catch (NSException *exception) { state(program).failed = YES; fail(error, error_capacity, exception.reason); return false; } }
}

bool indexed_ane_read(IndexedANEProgram *program, size_t index, void *bytes, size_t count, char *error, size_t error_capacity) {
    IndexedANEState *owner = state(program);
    if (!owner || !bytes || index >= owner.outputs.count || count != owner.outputSizes[index].unsignedLongLongValue) { fail(error, error_capacity, @"Invalid private ANE output bounds"); return false; }
    IOSurfaceRef value = (__bridge IOSurfaceRef)owner.outputs[index];
    if (IOSurfaceLock(value, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) { fail(error, error_capacity, @"Private ANE output lock failed"); return false; }
    memcpy(bytes, IOSurfaceGetBaseAddress(value), count);
    IOSurfaceUnlock(value, kIOSurfaceLockReadOnly, NULL);
    return true;
}

void indexed_ane_free(IndexedANEProgram *program) { if (program) { id owner = CFBridgingRelease(program); (void)owner; } }

void *indexed_ane_lock_output(IndexedANEProgram *program, size_t index, size_t count, char *error, size_t error_capacity) {
    IndexedANEState *owner = state(program);
    if (!owner || index >= owner.outputs.count || count != owner.outputSizes[index].unsignedLongLongValue) {
        fail(error, error_capacity, @"Invalid private ANE output borrow bounds"); return NULL;
    }
    IOSurfaceRef value = (__bridge IOSurfaceRef)owner.outputs[index];
    if (IOSurfaceLock(value, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) {
        fail(error, error_capacity, @"Private ANE output borrow lock failed"); return NULL;
    }
    return IOSurfaceGetBaseAddress(value);
}

void indexed_ane_unlock_output(IndexedANEProgram *program, size_t index) {
    IndexedANEState *owner = state(program);
    if (owner && index < owner.outputs.count) IOSurfaceUnlock((__bridge IOSurfaceRef)owner.outputs[index], kIOSurfaceLockReadOnly, NULL);
}

void *indexed_ane_retain_output_surface(IndexedANEProgram *program, size_t index) {
    IndexedANEState *owner = state(program);
    if (!owner || index >= owner.outputs.count) return NULL;
    return (void *)CFRetain((__bridge CFTypeRef)owner.outputs[index]);
}

void indexed_ane_release_surface(void *value) { if (value) CFRelease((CFTypeRef)value); }
