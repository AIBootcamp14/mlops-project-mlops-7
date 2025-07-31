import React, { useEffect, useState } from "react";
import { Card, CardContent } from "../../../../components/ui/card";

import img01 from "../../../../assets/01.png";
import img03 from "../../../../assets/03.png";
import img04 from "../../../../assets/04.png";
import img05 from "../../../../assets/05.png";

type Product = {
    id: number;
    imageUrl: string;
    price: string;
    rating: number;
};

export const RecommendedProductsSection = (): JSX.Element => {
    const [username, setUsername] = useState<string>("");
    const [recommendedProducts, setRecommendedProducts] = useState<Product[]>([]);

    const slideImages = [img01, img03, img04, img05];
    const [currentSlide, setCurrentSlide] = useState(0);

    useEffect(() => {
        fetch("/api/user")
            .then((res) => res.json())
            .then((data) => setUsername(data.user.name))
            .catch(console.error);

        fetch("/api/recommended")
            .then((res) => res.json())
            .then((data) => setRecommendedProducts(data))
            .catch(console.error);
    }, []);

    // 슬라이드 전환 타이머
    useEffect(() => {
        const interval = setInterval(() => {
            setCurrentSlide((prev) => (prev + 1) % slideImages.length);
        }, 3000);
        return () => clearInterval(interval);
    }, [slideImages.length]);

    return (
        <section className="w-full max-w-[990px] mx-auto my-8">
            {/* 유저 환영 메시지 */}
            <div className="text-xl font-semibold mb-4 text-[#333]"></div>

            {/* 배너 이미지 슬라이드 */}
            <Card className="w-full h-[350px] mb-8 overflow-hidden relative">
                <CardContent className="p-0 h-full relative">
                    {slideImages.map((img, index) => (
                        <img
                            key={index}
                            src={img}
                            alt={`Slide ${index + 1}`}
                            className={`
                                absolute inset-0 w-full h-full object-cover object-center
                                transition-opacity duration-700 ease-in-out
                                ${index === currentSlide ? "opacity-100 z-10" : "opacity-0 z-0"}
                            `}

                        />
                    ))}
                </CardContent>
            </Card>

            {/* 추천 상품 섹션 */}
            <div className="w-full">
                <div className="mb-6">
                    <img
                        className="h-9 w-auto"
                        alt="Recommendation text"
                        src="https://c.animaapp.com/mdn2wienEugbeI/img/recommendation-text.png"
                    />
                </div>

                <div className="grid grid-cols-4 gap-6">
                    {recommendedProducts.map((product) => (
                        <div key={product.id} className="flex flex-col">
                            <Card className="w-full mb-2">
                                <CardContent className="p-0">
                                    <div className="relative w-full h-[221px]">
                                        <img
                                            src={product.imageUrl}
                                            alt="상품 이미지"
                                            className="w-full h-full object-cover"
                                        />
                                    </div>
                                </CardContent>
                            </Card>
                            <div className="text-[#ff0004] text-base leading-snug">
                                {product.price}
                                <br />
                                ⭐ {product.rating} / 5.0
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
};
