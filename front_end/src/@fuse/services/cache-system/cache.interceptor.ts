// caching.interceptor.ts
import { inject, Injectable } from '@angular/core';
import { HttpEvent, HttpRequest, HttpHandlerFn, HttpResponse } from '@angular/common/http';
import { Observable, of, tap } from 'rxjs';
import { CacheService } from './cache.service';

export const cachingInterceptor = (
  req: HttpRequest<unknown>,
  next: HttpHandlerFn
): Observable<HttpEvent<unknown>> => {

  const cacheService = inject(CacheService);

  // List of specific URLs or partial paths to cache
  const cachedUrls = [
    '/api/settings',
    '^/api/v1/transductive/history/\\d+$'
  ];

  const shouldCache = cachedUrls.some(pattern => {
    // 'g' for global, 'i' for case-insensitive (optional)
    const regex = new RegExp(pattern, 'i');
    return regex.test(req.url);
  });

  if (!shouldCache) {
    return next(req);
  }

  // // Only cache GET requests
  // if (req.method !== 'GET') {
  //   return next(req);
  // }

  const cachedResponse = cacheService.get(req.urlWithParams);
  if (cachedResponse) {
    // Return cached response as an observable
    return of(cachedResponse);
  }

  // If not cached, send the request to the server and store the response
  return next(req).pipe(
    tap(event => {
      if (event instanceof HttpResponse) {
        cacheService.put(req.urlWithParams, event);
      }
    })
  );
};
