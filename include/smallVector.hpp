//===- llvm/ADT/SmallVector.h - 'Normally small' vectors --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <ostream>
#include <type_traits>
#include <utility>

namespace microtorch {

template <class Size_T>
class SmallVectorBase {
 protected:
  void* BeginX;
  Size_T Size = 0, Capacity;

  // The maximum value of the Size_T used.
  static constexpr size_t SizeTypeMax() {
    return std::numeric_limits<Size_T>::max();
  }

  SmallVectorBase(void* FirstEl, size_t TotalCapacity)
      : BeginX(FirstEl), Capacity(TotalCapacity) {}

  // This is a helper for grow() that's out of line to reduce code
  // duplication. This function will report a fatal error if it can't grow at
  // least to MinSize.
  void* mallocForGrow(size_t MinSize, size_t TSize, size_t& NewCapacity);

  // This is an implementation of the grow() method which only works
  // on POD-like data types and is out of line to reduce code duplication.
  // This function will report a fatal error if it cannot increase capacity.
  void grow_pod(void* FirstEl, size_t MinSize, size_t TSize);

 public:
  SmallVectorBase() = delete;
  size_t size() const { return Size; }
  size_t capacity() const { return Capacity; }
  bool empty() const { return !Size; }

  // Set the array size to N, which the current array must have enough
  // capacity for.
  void set_size(size_t N) {
    assert(N <= capacity());
    Size = N;
  }
};

template <class T>
using SmallVectorSizeType =
    typename std::conditional_t<sizeof(T) < 4 && sizeof(void*) >= 8, uint64_t,
                                uint32_t>;

// Figure out the offset of the first element.
template <class T, typename = void>
struct SmallVectorAlignmentAndSize {
  alignas(SmallVectorBase<SmallVectorSizeType<T>>) char Base[sizeof(
      SmallVectorBase<SmallVectorSizeType<T>>)];
  alignas(T) char FirstEl[sizeof(T)];
};

// This is the part of SmallVectorTemplateBase which does not depend on whether
// the type T is a POD. The extra dummy template argument is used by ArrayRef
// to avoid unnecessarily requiring T to be complete.
template <typename T, typename = void>
class SmallVectorTemplateCommon
    : public SmallVectorBase<SmallVectorSizeType<T>> {
  using Base = SmallVectorBase<SmallVectorSizeType<T>>;

  // Find the address of the first element.  For this pointer math to be valid
  // with small-size of 0 for T with lots of alignment, it's important that
  // SmallVectorStorage is properly-aligned even for small-size of 0.
  void* getFirstEl() const {
    return const_cast<void*>(reinterpret_cast<const void*>(
        reinterpret_cast<const char*>(this) +
        offsetof(SmallVectorAlignmentAndSize<T>, FirstEl)));
  }
  // Space after 'FirstEl' is clobbered, do not add any instance vars after it.

 protected:
  SmallVectorTemplateCommon(size_t Size) : Base(getFirstEl(), Size) {}

  void grow_pod(size_t MinSize, size_t TSize) {
    Base::grow_pod(getFirstEl(), MinSize, TSize);
  }

  // Return true if this is a smallvector which has not had dynamic
  // memory allocated for it.
  bool isSmall() const { return this->BeginX == getFirstEl(); }

  // Put this vector in a state of being small.
  void resetToSmall() {
    this->BeginX = getFirstEl();
    this->Size = this->Capacity =
        0;  // FIXME: Setting Capacity to 0 is suspect.
  }

  // Return true if V is an internal reference to the given range.
  bool isReferenceToRange(const void* V, const void* First,
                          const void* Last) const {
    // Use std::less to avoid UB.
    std::less<> LessThan;
    return !LessThan(V, First) && LessThan(V, Last);
  }

  // Return true if V is an internal reference to this vector.
  bool isReferenceToStorage(const void* V) const {
    return isReferenceToRange(V, this->begin(), this->end());
  }

  // Return true if First and Last form a valid (possibly empty) range in this
  // vector's storage.
  bool isRangeInStorage(const void* First, const void* Last) const {
    // Use std::less to avoid UB.
    std::less<> LessThan;
    return !LessThan(First, this->begin()) && !LessThan(Last, First) &&
           !LessThan(this->end(), Last);
  }

  // Return true unless Elt will be invalidated by resizing the vector to
  // NewSize.
  bool isSafeToReferenceAfterResize(const void* Elt, size_t NewSize) {
    // Past the end.
    if (!isReferenceToStorage(Elt)) return true;

    // Return false if Elt will be destroyed by shrinking.
    if (NewSize <= this->size()) return Elt < this->begin() + NewSize;

    // Return false if we need to grow.
    return NewSize <= this->capacity();
  }

  // Check whether Elt will be invalidated by resizing the vector to NewSize.
  void assertSafeToReferenceAfterResize(const void* Elt, size_t NewSize) {
    (void)Elt;      // Suppress unused variable warning
    (void)NewSize;  // Suppress unused variable warning
    assert(isSafeToReferenceAfterResize(Elt, NewSize) &&
           "Attempting to reference an element of the vector in an operation "
           "that invalidates it");
  }

  // Check whether Elt will be invalidated by increasing the size of the
  // vector by N.
  void assertSafeToAdd(const void* Elt, size_t N = 1) {
    this->assertSafeToReferenceAfterResize(Elt, this->size() + N);
  }

  // Check whether any part of the range will be invalidated by clearing.
  void assertSafeToReferenceAfterClear(const T* From, const T* To) {
    if (From == To) return;
    this->assertSafeToReferenceAfterResize(From, 0);
    this->assertSafeToReferenceAfterResize(To - 1, 0);
  }
  template <
      class IterType,
      std::enable_if_t<!std::is_same<std::remove_const_t<IterType>, T*>::value,
                       bool> = false>
  void assertSafeToReferenceAfterClear(IterType, IterType) {}

  // Check whether any part of the range will be invalidated by growing.
  void assertSafeToAddRange(const T* From, const T* To) {
    if (From == To) return;
    this->assertSafeToAdd(From, To - From);
    this->assertSafeToAdd(To - 1, To - From);
  }
  template <
      class IterType,
      std::enable_if_t<!std::is_same<std::remove_const_t<IterType>, T*>::value,
                       bool> = false>
  void assertSafeToAddRange(IterType, IterType) {}

  // Reserve enough space to add one element, and return the updated element
  // pointer in case it was a reference to the storage.
  template <class U>
  static const T* reserveForParamAndGetAddressImpl(U* This, const T& Elt,
                                                   size_t N) {
    size_t NewSize = This->size() + N;
    if (NewSize <= This->capacity()) return &Elt;

    bool ReferencesStorage = false;
    int64_t Index = -1;
    if (!U::TakesParamByValue) {
      if (This->isReferenceToStorage(&Elt)) {
        ReferencesStorage = true;
        Index = &Elt - This->begin();
      }
    }
    This->grow(NewSize);
    return ReferencesStorage ? This->begin() + Index : &Elt;
  }

 public:
  using Base::capacity;
  using Base::empty;
  using Base::size;

  // forward iterator creation methods.
  T* begin() { return static_cast<T*>(this->BeginX); }
  const T* begin() const { return static_cast<const T*>(this->BeginX); }
  T* end() { return begin() + size(); }
  const T* end() const { return begin() + size(); }

  // reverse iterator creation methods.
  std::reverse_iterator<T*> rbegin() { return std::reverse_iterator<T*>(end()); }
  std::reverse_iterator<const T*> rbegin() const {
    return std::reverse_iterator<const T*>(end());
  }
  std::reverse_iterator<T*> rend() { return std::reverse_iterator<T*>(begin()); }
  std::reverse_iterator<const T*> rend() const {
    return std::reverse_iterator<const T*>(begin());
  }

  size_t size_in_bytes() const { return size() * sizeof(T); }
  size_t max_size() const {
    return std::min(this->SizeTypeMax(), size_t(-1) / sizeof(T));
  }

  // Return a pointer to the vector's buffer, even if empty().
  T* data() { return begin(); }
  const T* data() const { return begin(); }

  T& at(size_t idx) {
    assert(idx < size());
    return begin()[idx];
  }
  const T& at(size_t idx) const {
    assert(idx < size());
    return begin()[idx];
  }
  T& operator[](size_t idx) {
    assert(idx < size());
    return begin()[idx];
  }
  const T& operator[](size_t idx) const {
    assert(idx < size());
    return begin()[idx];
  }

  T& front() {
    assert(!empty());
    return begin()[0];
  }
  const T& front() const {
    assert(!empty());
    return begin()[0];
  }

  T& back() {
    assert(!empty());
    return end()[-1];
  }
  const T& back() const {
    assert(!empty());
    return end()[-1];
  }
};

// SmallVectorTemplateBase<TriviallyCopyable = false> - This is where we put
// method implementations that are designed to work with non-trivial T's.
///
// We approximate is_trivially_copyable with trivial move/copy construction and
// trivial destruction. While the standard doesn't specify that you're allowed
// copy these types with memcpy, there is no way for the type to observe this.
// This catches the important case of std::pair<POD, POD>, which is not
// trivially assignable.
///
template <typename T, bool = (std::is_trivially_copy_constructible<T>::value) &&
                             (std::is_trivially_move_constructible<T>::value) &&
                             std::is_trivially_destructible<T>::value>
class SmallVectorTemplateBase : public SmallVectorTemplateCommon<T> {
  friend class SmallVectorTemplateCommon<T>;

 protected:
  static constexpr bool TakesParamByValue = false;
  using ValueParamT = const T&;

  SmallVectorTemplateBase(size_t Size) : SmallVectorTemplateCommon<T>(Size) {}

  static void destroy_range(T* S, T* E) {
    while (S != E) {
      --E;
      E->~T();
    }
  }

  // Move the range [I, E) into the uninitialized memory starting with "Dest",
  // constructing elements as needed.
  template <typename It1, typename It2>
  static void uninitialized_move(It1 I, It1 E, It2 Dest) {
    std::uninitialized_copy(std::make_move_iterator(I),
                            std::make_move_iterator(E), Dest);
  }

  // Copy the range [I, E) onto the uninitialized memory starting with "Dest",
  // constructing elements as needed.
  template <typename It1, typename It2>
  static void uninitialized_copy(It1 I, It1 E, It2 Dest) {
    std::uninitialized_copy(I, E, Dest);
  }

  // Grow the allocated memory (without initializing new elements), doubling
  // the size of the allocated memory. Guarantees space for at least one more
  // element, or MinSize more elements if specified.
  void grow(size_t MinSize = 0);

  // Create a new allocation big enough for MinSize and pass back its size
  // in NewCapacity. This is the first section of grow().
  T* mallocForGrow(size_t MinSize, size_t& NewCapacity) {
    return static_cast<T*>(
        SmallVectorBase<SmallVectorSizeType<T>>::mallocForGrow(
            MinSize, sizeof(T), NewCapacity));
  }

  // Move existing elements over to the new allocation NewElts, the middle
  // section of grow().
  void moveElementsForGrow(T* NewElts);

  // Transfer ownership of the allocation, finishing up grow().
  void takeAllocationForGrow(T* NewElts, size_t NewCapacity);

  // Reserve enough space to add one element, and return the updated element
  // pointer in case it was a reference to the storage.
  const T* reserveForParamAndGetAddress(const T& Elt, size_t N = 1) {
    return this->reserveForParamAndGetAddressImpl(this, Elt, N);
  }

  // Reserve enough space to add one element, and return the updated element
  // pointer in case it was a reference to the storage.
  T* reserveForParamAndGetAddress(T& Elt, size_t N = 1) {
    return const_cast<T*>(this->reserveForParamAndGetAddressImpl(this, Elt, N));
  }

  void growAndAssign(size_t NumElts, const T& Elt) {
    // Grow manually in case Elt is an internal reference.
    size_t NewCapacity = 0;
    T* NewElts = mallocForGrow(NumElts, NewCapacity);
    std::uninitialized_fill_n(NewElts, NumElts, Elt);
    this->destroy_range(this->begin(), this->end());
    takeAllocationForGrow(NewElts, NewCapacity);
    this->set_size(NumElts);
  }

  template <typename... ArgTypes>
  T& growAndEmplaceBack(ArgTypes&&... Args) {
    // Grow manually in case one of Args is an internal reference.
    size_t NewCapacity = 0;
    T* NewElts = mallocForGrow(0, NewCapacity);
    ::new ((void*)(NewElts + this->size())) T(std::forward<ArgTypes>(Args)...);
    moveElementsForGrow(NewElts);
    takeAllocationForGrow(NewElts, NewCapacity);
    this->set_size(this->size() + 1);
    return this->back();
  }

 public:
  void push_back(const T& Elt) {
    const T* EltPtr = reserveForParamAndGetAddress(Elt);
    ::new ((void*)this->end()) T(*EltPtr);
    this->set_size(this->size() + 1);
  }

  void push_back(T&& Elt) {
    T* EltPtr = reserveForParamAndGetAddress(Elt);
    ::new ((void*)this->end()) T(::std::move(*EltPtr));
    this->set_size(this->size() + 1);
  }

  void pop_back() {
    this->set_size(this->size() - 1);
    this->end()->~T();
  }
};

// Define this out-of-line to dissuade the C++ compiler from inlining it.
template <typename T, bool TriviallyCopyable>
void SmallVectorTemplateBase<T, TriviallyCopyable>::grow(size_t MinSize) {
  size_t NewCapacity = 0;
  T* NewElts = mallocForGrow(MinSize, NewCapacity);
  moveElementsForGrow(NewElts);
  takeAllocationForGrow(NewElts, NewCapacity);
}

// Define this out-of-line to dissuade the C++ compiler from inlining it.
template <typename T, bool TriviallyCopyable>
void SmallVectorTemplateBase<T, TriviallyCopyable>::moveElementsForGrow(
    T* NewElts) {
  // Move the elements over.
  this->uninitialized_move(this->begin(), this->end(), NewElts);

  // Destroy the original elements.
  destroy_range(this->begin(), this->end());
}

// Define this out-of-line to dissuade the C++ compiler from inlining it.
template <typename T, bool TriviallyCopyable>
void SmallVectorTemplateBase<T, TriviallyCopyable>::takeAllocationForGrow(
    T* NewElts, size_t NewCapacity) {
  // If this wasn't grown from the inline copy, deallocate the old space.
  if (!this->isSmall()) free(this->begin());

  this->BeginX = NewElts;
  this->Capacity = NewCapacity;
}

// SmallVectorTemplateBase<TriviallyCopyable = true> - This is where we put
// method implementations that are designed to work with trivially copyable
// T's. This allows using memcpy in place of copy/move construction and
// skipping destruction.
template <typename T>
class SmallVectorTemplateBase<T, true> : public SmallVectorTemplateCommon<T> {
  friend class SmallVectorTemplateCommon<T>;

 protected:
  // True if it's cheap enough to take parameters by value. Doing so avoids
  // overhead related to mitigations for reference invalidation.
  static constexpr bool TakesParamByValue = sizeof(T) <= 2 * sizeof(void*);

  // Either const T& or T, depending on whether it's cheap enough to take
  // parameters by value.
  using ValueParamT =
      typename std::conditional_t<TakesParamByValue, T, const T&>;

  SmallVectorTemplateBase(size_t Size) : SmallVectorTemplateCommon<T>(Size) {}

  // No need to do a destroy loop for POD's.
  static void destroy_range(T*, T*) {}

  // Move the range [I, E) onto the uninitialized memory
  // starting with "Dest", constructing elements into it as needed.
  template <typename It1, typename It2>
  static void uninitialized_move(It1 I, It1 E, It2 Dest) {
    // Just do a copy.
    uninitialized_copy(I, E, Dest);
  }

  // Copy the range [I, E) onto the uninitialized memory
  // starting with "Dest", constructing elements into it as needed.
  template <typename It1, typename It2>
  static void uninitialized_copy(It1 I, It1 E, It2 Dest) {
    // Arbitrary iterator types; just use the basic implementation.
    std::uninitialized_copy(I, E, Dest);
  }

  // Copy the range [I, E) onto the uninitialized memory
  // starting with "Dest", constructing elements into it as needed.
  template <typename T1, typename T2>
  static void uninitialized_copy(
      T1* I, T1* E, T2* Dest,
      std::enable_if_t<std::is_same<typename std::remove_const<T1>::type,
                                    T2>::value>* = nullptr) {
    // Use memcpy for PODs iterated by pointers (which includes SmallVector
    // iterators): std::uninitialized_copy optimizes to memmove, but we can
    // use memcpy here. Note that I and E are iterators and thus might be
    // invalid for memcpy if they are equal.
    if (I != E) memcpy(reinterpret_cast<void*>(Dest), I, (E - I) * sizeof(T));
  }

  // Double the size of the allocated memory, guaranteeing space for at
  // least one more element or MinSize if specified.
  void grow(size_t MinSize = 0) { this->grow_pod(MinSize, sizeof(T)); }

  // Reserve enough space to add one element, and return the updated element
  // pointer in case it was a reference to the storage.
  const T* reserveForParamAndGetAddress(const T& Elt, size_t N = 1) {
    return this->reserveForParamAndGetAddressImpl(this, Elt, N);
  }

  // Reserve enough space to add one element, and return the updated element
  // pointer in case it was a reference to the storage.
  T* reserveForParamAndGetAddress(T& Elt, size_t N = 1) {
    return const_cast<T*>(this->reserveForParamAndGetAddressImpl(this, Elt, N));
  }

  void growAndAssign(size_t NumElts, T Elt) {
    // Elt has been copied in case it's an internal reference, side-stepping
    // reference invalidation problems without losing the realloc optimization.
    this->set_size(0);
    this->grow(NumElts);
    std::uninitialized_fill_n(this->begin(), NumElts, Elt);
    this->set_size(NumElts);
  }

  template <typename... ArgTypes>
  T& growAndEmplaceBack(ArgTypes&&... Args) {
    // Use push_back with a copy in case Args has an internal reference,
    // side-stepping reference invalidation problems without losing the realloc
    // optimization.
    push_back(T(std::forward<ArgTypes>(Args)...));
    return this->back();
  }

 public:
  void push_back(ValueParamT Elt) {
    const T* EltPtr = reserveForParamAndGetAddress(Elt);
    memcpy(reinterpret_cast<void*>(this->end()), EltPtr, sizeof(T));
    this->set_size(this->size() + 1);
  }

  void pop_back() { this->set_size(this->size() - 1); }
};

// This class consists of common code factored out of the SmallVector class to
// reduce code duplication based on the SmallVector 'N' template parameter.
template <typename T>
class SmallVectorImpl : public SmallVectorTemplateBase<T> {
 protected:
  using SmallVectorTemplateBase<T>::TakesParamByValue;
  using ValueParamT = typename SmallVectorTemplateBase<T>::ValueParamT;

  // Default constructor - Initialize to empty.
  explicit SmallVectorImpl(unsigned N) : SmallVectorTemplateBase<T>(N) {}

 public:
  SmallVectorImpl(const SmallVectorImpl&) = delete;

  ~SmallVectorImpl() {
    // Subclass has already destructed this vector's elements.
    // If this wasn't grown from the inline copy, deallocate the old space.
    if (!this->isSmall()) free(this->begin());
  }

  void clear() {
    this->destroy_range(this->begin(), this->end());
    this->Size = 0;
  }

 private:
  template <bool ForOverwrite>
  void resizeImpl(size_t N) {
    if (N < this->size()) {
      this->pop_back_n(this->size() - N);
    } else if (N > this->size()) {
      this->reserve(N);
      for (auto I = this->end(), E = this->begin() + N; I != E; ++I)
        if (ForOverwrite)
          new (&*I) T;
        else
          new (&*I) T();
      this->set_size(N);
    }
  }

 public:
  void resize(size_t N) { resizeImpl<false>(N); }

  void resize(size_t N, ValueParamT NV) {
    if (N == this->size()) return;

    if (N < this->size()) {
      this->pop_back_n(this->size() - N);
      return;
    }

    // N > this->size(). Defer to append.
    this->append(N - this->size(), NV);
  }

  void reserve(size_t N) {
    if (this->capacity() < N) this->grow(N);
  }

  void pop_back_n(size_t NumItems) {
    assert(this->size() >= NumItems);
    this->destroy_range(this->end() - NumItems, this->end());
    this->set_size(this->size() - NumItems);
  }

  // Add the specified range to the end of the SmallVector.
  template <typename in_iter,
            typename = std::enable_if_t<std::is_convertible_v<
                typename std::iterator_traits<in_iter>::iterator_category,
                std::input_iterator_tag>>>
  void append(in_iter in_start, in_iter in_end) {
    this->assertSafeToAddRange(in_start, in_end);
    size_t NumInputs = std::distance(in_start, in_end);
    this->reserve(this->size() + NumInputs);
    this->uninitialized_copy(in_start, in_end, this->end());
    this->set_size(this->size() + NumInputs);
  }

  // Append NumInputs copies of Elt to the end.
  void append(size_t NumInputs, ValueParamT Elt) {
    const T* EltPtr = this->reserveForParamAndGetAddress(Elt, NumInputs);
    std::uninitialized_fill_n(this->end(), NumInputs, *EltPtr);
    this->set_size(this->size() + NumInputs);
  }

  void append(std::initializer_list<T> IL) { append(IL.begin(), IL.end()); }

  void append(const SmallVectorImpl& RHS) { append(RHS.begin(), RHS.end()); }

  void assign(size_t NumElts, ValueParamT Elt) {
    // Note that Elt could be an internal reference.
    if (NumElts > this->capacity()) {
      this->growAndAssign(NumElts, Elt);
      return;
    }

    // Assign over existing elements.
    std::fill_n(this->begin(), std::min(NumElts, this->size()), Elt);
    if (NumElts > this->size())
      std::uninitialized_fill_n(this->end(), NumElts - this->size(), Elt);
    else if (NumElts < this->size())
      this->destroy_range(this->begin() + NumElts, this->end());
    this->set_size(NumElts);
  }

  // FIXME: Consider assigning over existing elements, rather than clearing &
  // re-initializing them - for all assign(...) variants.
  template <typename in_iter,
            typename = std::enable_if_t<std::is_convertible_v<
                typename std::iterator_traits<in_iter>::iterator_category,
                std::input_iterator_tag>>>
  void assign(in_iter in_start, in_iter in_end) {
    this->assertSafeToReferenceAfterClear(in_start, in_end);
    clear();
    append(in_start, in_end);
  }

  void assign(std::initializer_list<T> IL) {
    clear();
    append(IL);
  }

  void assign(const SmallVectorImpl& RHS) { assign(RHS.begin(), RHS.end()); }

  template <typename... ArgTypes>
  T& emplace_back(ArgTypes&&... Args) {
    if (this->size() >= this->capacity())
      return this->growAndEmplaceBack(std::forward<ArgTypes>(Args)...);

    ::new ((void*)this->end()) T(std::forward<ArgTypes>(Args)...);
    this->set_size(this->size() + 1);
    return this->back();
  }

  SmallVectorImpl& operator=(const SmallVectorImpl& RHS) {
    // Avoid self-assignment.
    if (this == &RHS) return *this;

    // If we already have sufficient space, assign the common elements, then
    // destroy any excess.
    size_t RHSSize = RHS.size();
    size_t CurSize = this->size();
    if (CurSize >= RHSSize) {
      // Assign common elements.
      T* NewEnd;
      if (RHSSize)
        NewEnd = std::copy(RHS.begin(), RHS.begin() + RHSSize, this->begin());
      else
        NewEnd = this->begin();

      // Destroy excess elements.
      this->destroy_range(NewEnd, this->end());

      // Trim.
      this->set_size(RHSSize);
      return *this;
    }

    // If we have to grow to have enough elements, destroy the current elements.
    // This allows us to avoid copying them during the grow.
    // FIXME: don't do this if they're efficiently moveable.
    if (this->capacity() < RHSSize) {
      // Destroy current elements.
      this->clear();
      CurSize = 0;
      this->grow(RHSSize);
    } else if (CurSize) {
      // Otherwise, use assignment for the already-constructed elements.
      std::copy(RHS.begin(), RHS.begin() + CurSize, this->begin());
    }

    // Copy construct the new elements in place.
    this->uninitialized_copy(RHS.begin() + CurSize, RHS.end(),
                             this->begin() + CurSize);

    // Set end.
    this->set_size(RHSSize);
    return *this;
  }

  SmallVectorImpl& operator=(SmallVectorImpl&& RHS) noexcept(
      std::is_nothrow_move_constructible_v<T> &&
      std::is_nothrow_destructible_v<T>) {
    // Avoid self-assignment.
    if (this == &RHS) return *this;

    // If the RHS isn't small, clear this vector and then steal its buffer.
    if (!RHS.isSmall()) {
      this->destroy_range(this->begin(), this->end());
      if (!this->isSmall()) free(this->begin());
      this->BeginX = RHS.BeginX;
      this->Size = RHS.Size;
      this->Capacity = RHS.Capacity;
      RHS.resetToSmall();
      return *this;
    }

    // If we already have sufficient space, assign the common elements, then
    // destroy any excess.
    size_t RHSSize = RHS.size();
    size_t CurSize = this->size();
    if (CurSize >= RHSSize) {
      // Assign common elements.
      T* NewEnd = this->begin();
      if (RHSSize) NewEnd = std::move(RHS.begin(), RHS.end(), NewEnd);

      // Destroy excess elements and trim the bounds.
      this->destroy_range(NewEnd, this->end());
      this->set_size(RHSSize);

      // Clear the RHS.
      RHS.clear();

      return *this;
    }

    // If we have to grow to have enough elements, destroy the current elements.
    // This allows us to avoid copying them during the grow.
    // FIXME: this may not actually make any sense if we can efficiently move
    // elements.
    if (this->capacity() < RHSSize) {
      // Destroy current elements.
      this->clear();
      CurSize = 0;
      this->grow(RHSSize);
    } else if (CurSize) {
      // Otherwise, use assignment for the already-constructed elements.
      std::move(RHS.begin(), RHS.begin() + CurSize, this->begin());
    }

    // Move-construct the new elements in place.
    this->uninitialized_move(RHS.begin() + CurSize, RHS.end(),
                             this->begin() + CurSize);

    // Set end.
    this->set_size(RHSSize);

    RHS.clear();
    return *this;
  }
};

// Inline stack storage for the SmallVector elements.
// This will be visited directly by SmallVectorImpl.
template <typename T, unsigned N>
struct SmallVectorStorage {
  alignas(T) char elements[N * sizeof(T)];
};

// We need the storage to be properly aligned even for small-size of 0.
// This is designed for `SmallVectorTemplateCommon::getFirstEl()`
template <typename T>
struct alignas(T) SmallVectorStorage<T, 0> {};

// Forward declaration for `calculateDefaultElements`
template <typename T, unsigned N>
class SmallVector;

// Calculating the default number of inline elements for `SmallVector<T>`.
template <typename T>
constexpr size_t calculateDefaultElements() {
  // The default number of inlined elements for `SmallVector<T>`.
  constexpr size_t kPreferredSmallVectorSizeof = 64;
  static_assert(
      sizeof(T) <= 256,
      "You are trying to use a default number of inlined elements for "
      "`SmallVector<T>` but `sizeof(T)` is really big! Please use an "
      "explicit number of inlined elements with `SmallVector<T, N>` to make "
      "sure you really want that much inline storage.");

  // Discount the size of the header when calculating the maximum inline bytes.
  constexpr size_t PreferredInlineBytes =
      kPreferredSmallVectorSizeof - sizeof(SmallVector<T, 0>);
  constexpr size_t NumElementsThatFit = PreferredInlineBytes / sizeof(T);
  // At least 1
  return NumElementsThatFit == 0 ? 1 : NumElementsThatFit;
};

// This is a variable-sized vector, optimized for the small array.
// It contains some number of elements in-place,
// which allows it to avoid heap allocation when the actual number of
// elements is below that threshold.
// Note: This does not attempt to be exception safe.
template <typename T,
          unsigned N = calculateDefaultElements<T>()>
class SmallVector : public SmallVectorImpl<T>, SmallVectorStorage<T, N> {
 public:
  SmallVector() : SmallVectorImpl<T>(N) {}

  ~SmallVector() { this->destroy_range(this->begin(), this->end()); }

  explicit SmallVector(size_t Size, const T& Value = T())
      : SmallVectorImpl<T>(N) {
    this->assign(Size, Value);
  }

  template <typename IterType,
            typename = std::enable_if_t<std::is_convertible_v<
                typename std::iterator_traits<IterType>::iterator_category,
                std::input_iterator_tag>>>
  SmallVector(IterType S, IterType E) : SmallVectorImpl<T>(N) {
    this->append(S, E);
  }

  // The enable_if restricts Container to types that have a .begin() and
  // .end() that return valid input iterators.
  template <
      typename Container,
      std::enable_if_t<
          std::is_convertible_v<typename std::iterator_traits<
                                    decltype(std::declval<Container>()
                                                 .begin())>::iterator_category,
                                std::input_iterator_tag> &&
              std::is_convertible_v<
                  typename std::iterator_traits<
                      decltype(std::declval<Container>()
                                   .end())>::iterator_category,
                  std::input_iterator_tag>,
          int> = 0>
  explicit SmallVector(Container&& c) : SmallVectorImpl<T>(N) {
    this->append(c.begin(), c.end());
  }

  SmallVector(std::initializer_list<T> IL) : SmallVectorImpl<T>(N) {
    this->assign(IL);
  }

  SmallVector(const SmallVector& RHS) : SmallVectorImpl<T>(N) {
    if (!RHS.empty()) SmallVectorImpl<T>::operator=(RHS);
  }

  SmallVector& operator=(const SmallVector& RHS) {
    SmallVectorImpl<T>::operator=(RHS);
    return *this;
  }

  SmallVector(SmallVector&& RHS) noexcept(
      std::is_nothrow_move_assignable_v<SmallVectorImpl<T>>)
      : SmallVectorImpl<T>(N) {
    if (!RHS.empty()) SmallVectorImpl<T>::operator=(::std::move(RHS));
  }

  template <
      typename Container,
      std::enable_if_t<
          std::is_convertible_v<typename std::iterator_traits<
                                    decltype(std::declval<Container>()
                                                 .begin())>::iterator_category,
                                std::input_iterator_tag> &&
              std::is_convertible_v<
                  typename std::iterator_traits<
                      decltype(std::declval<Container>()
                                   .end())>::iterator_category,
                  std::input_iterator_tag>,
          int> = 0>
  SmallVector& operator=(const Container& RHS) {
    this->assign(RHS.begin(), RHS.end());
    return *this;
  }

  template <
      typename Container,
      std::enable_if_t<
          std::is_convertible_v<typename std::iterator_traits<
                                    decltype(std::declval<Container>()
                                                 .begin())>::iterator_category,
                                std::input_iterator_tag> &&
              std::is_convertible_v<
                  typename std::iterator_traits<
                      decltype(std::declval<Container>()
                                   .end())>::iterator_category,
                  std::input_iterator_tag>,
          int> = 0>
  SmallVector& operator=(Container&& C) {
    this->assign(C.begin(), C.end());
    return *this;
  }

  SmallVector& operator=(SmallVector&& RHS) noexcept(
      std::is_nothrow_move_assignable_v<SmallVectorImpl<T>>) {
    SmallVectorImpl<T>::operator=(::std::move(RHS));
    return *this;
  }

  SmallVector& operator=(std::initializer_list<T> IL) {
    this->assign(IL);
    return *this;
  }
};

}  // end namespace microtorch
